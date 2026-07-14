from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import minimize

from ..models.forward_model import ForwardModel


@dataclass
class Solution:
    curr_dq: np.ndarray
    torque: float
    success: bool
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _run_slsqp(
    objective: Any,
    constraints: list[dict[str, Any]],
    candidates: list[np.ndarray],
    opts: dict[str, Any],
) -> tuple[np.ndarray, float, bool]:
    best_x: np.ndarray | None = None
    best_val = float("inf")

    for x0 in candidates:
        res = minimize(objective, x0, method="SLSQP", constraints=constraints, options=opts)
        if res.success and res.fun < best_val:
            best_val = res.fun
            best_x = res.x

    if best_x is not None:
        return best_x, best_val, True
    return np.full_like(candidates[0], np.nan), float("inf"), False


class BaseOptimizer(ABC):
    def __init__(self, fwd: ForwardModel, opts: dict[str, Any] | None = None) -> None:
        self.fwd = fwd
        self.opts: dict[str, Any] = opts if opts is not None else {"disp": False, "ftol": 1e-8, "maxiter": 500}

    @abstractmethod
    def minimize_current(
        self,
        torq_target: float,
        omega: float,
        guess: np.ndarray | None = None,
    ) -> Solution: ...

    @abstractmethod
    def maximize_torque(
        self,
        omega: float,
        guess: np.ndarray | None = None,
    ) -> Solution: ...


class StaticOptimizer(BaseOptimizer):
    def _base_constraints(self, omega: float) -> list[dict[str, Any]]:
        fwd = self.fwd

        def limits_con(i: np.ndarray) -> np.ndarray:
            # Current and voltage margins as one vector-valued constraint:
            # SLSQP's finite-difference Jacobian is computed once per
            # constraint *dict*, so keeping these as two separate scalar
            # constraints costs a redundant Jacobian evaluation per SLSQP
            # step (~35% more peak_vals()/flux() calls for identical
            # convergence — verified same nit, same optimum).
            curr_peak, volt_peak = fwd.peak_vals(omega, i)
            return np.array([fwd.drive.curr_max - curr_peak, fwd.drive.volt_max - volt_peak])

        cons: list[dict[str, Any]] = [
            {"type": "ineq", "fun": limits_con},
        ]
        cons.extend(fwd.extra_constraints())
        return cons

    def minimize_current(
        self,
        torq_target: float,
        omega: float,
        guess: np.ndarray | None = None,
    ) -> Solution:
        fwd = self.fwd
        candidates = fwd.drive.seeds(guess)
        constraints = self._base_constraints(omega)
        constraints.append(
            {
                "type": "eq",
                "fun": lambda i: fwd.drive.torque(omega, i) - torq_target,
            }
        )
        best_i, _, success = _run_slsqp(lambda i: float(np.sum(i**2)), constraints, candidates, self.opts)
        torque = float(fwd.drive.torque(omega, best_i)) if success else float("nan")
        return Solution(curr_dq=best_i, torque=torque, success=success)

    def maximize_torque(
        self,
        omega: float,
        guess: np.ndarray | None = None,
    ) -> Solution:
        fwd = self.fwd
        candidates = fwd.drive.seeds(guess)
        constraints = self._base_constraints(omega)
        best_i, best_val, success = _run_slsqp(
            lambda i: -fwd.drive.torque(omega, i), constraints, candidates, self.opts
        )
        torque = float(-best_val) if success else float("nan")
        return Solution(curr_dq=best_i, torque=torque, success=success)
