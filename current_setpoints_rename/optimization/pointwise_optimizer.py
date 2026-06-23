from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize

from ..simulation import BaseTransform
from .models import BaseTorqueModel


class PointwiseOptimizer:
    """
    Instantaneous (per-θ) optimizer for fault-tolerant motor control.

    Each discrete angle θ is treated as an independent sub-problem: find
    the DQ current vector that maximises instantaneous torque subject to
    the phase current and voltage limits **at that specific θ**, plus the
    θ-dependent null-space equality returned by
    ``transform.get_extra_constraints_at_theta`` (for two-fault modes,
    ``N @ R(θ) @ i_s = 0``).  The rated torque is then the minimum of the
    resulting torque envelope across all sampled angles.

    Note: the per-θ phase-current constraints alone are **not** sufficient
    for two-fault modes. The dynamic null-space equality
    ``N @ R(θ) @ i_s = 0`` pins the αβ vector ``R(θ) @ i_s`` to
    ``col(C_red)`` at each θ, so the realized phase currents match ``i_s``
    and the dq torque ``T(i_s) = i_sᵀA·i_s + 2·bᵀ·i_s`` equals the realized
    torque. The static ``N @ i_s = 0`` (paper Eq. 23 / 34) only holds at
    θ=0; for a per-θ varying ``i_s`` it leaves the realized αβ off
    ``col(C_red)`` and the dq torque becomes non-physical (this is what the
    ``MotorOptimizer`` static scheme uses, where ``i_s`` is constant).
    """

    def __init__(
        self,
        model: BaseTorqueModel,
        n_theta: int = 360,
        opts: dict[str, Any] | None = None,
        n_random_starts: int = 0,
        random_seed: int = 0,
    ) -> None:
        """
        Args:
            model: Torque model used for objective evaluation.
            n_theta: Number of rotor-angle samples drawn from the transform's
                ``vec_theta`` for envelope computation.
            opts: SLSQP solver options dict.
            n_random_starts: Number of uniform-random seed points in
                ``[-curr_max, curr_max]^dim`` appended to ``model.get_candidates``
                at each θ. Zero reproduces the original 3-seed behavior; larger
                values mitigate SLSQP getting stuck in local optima at the cost
                of proportional runtime.
            random_seed: Seed for the random-start RNG (for reproducibility).
        """
        self.model = model
        self.n_theta = n_theta
        self.opts = opts if opts is not None else {"disp": False, "ftol": 1e-8, "maxiter": 500}
        self.n_random_starts = n_random_starts
        self._rng = np.random.default_rng(random_seed)

    def _candidates(self, curr_dq_guess: np.ndarray | None) -> list[np.ndarray]:
        candidates = self.model.get_candidates(curr_dq_guess)
        if self.n_random_starts > 0:
            curr_max = self.model.curr_max
            dim = self.model.n_phases - 1
            for _ in range(self.n_random_starts):
                candidates.append(self._rng.uniform(-curr_max, curr_max, dim))
        return candidates

    def _get_theta_indices(self, transform: BaseTransform) -> np.ndarray:
        n_t = transform.vec_theta.size
        return np.round(np.linspace(0, n_t - 1, self.n_theta)).astype(int)

    def _build_constraints_at_theta(
        self,
        theta_idx: int,
        omega: float,
        transform: BaseTransform,
    ) -> list[dict[str, Any]]:
        """
        Builds SLSQP constraints for a single rotor angle:

        - Phase-current inequalities ``|h_k(θ)·i_s| ≤ I_max`` for each
          active phase ``k`` (paper Eq. 10c / 20c / 31d / 42d).
        - Phase-voltage inequalities ``|h_k(θ)·(U·i_s + u)| ≤ V_max``
          (paper Eq. 10d / 20d / 31e / 42e). SVPWM zero-sequence
          injection is disabled.
        - Plus the θ-dependent null-space equality
          ``transform.get_extra_constraints_at_theta(θ)`` provides — for
          two-fault modes this is ``N @ R(θ) @ i_s = 0`` (the dynamic form
          of paper Eq. 23 / 34, required for the per-θ scheme so the
          realized αβ stays in ``col(C_red)`` and the dq torque equals the
          realized torque); for healthy and single-fault it is empty.

        Each ``|v| ≤ limit`` becomes two half-space inequalities:
        ``limit - v ≥ 0`` and ``limit + v ≥ 0``.
        """
        H_ph = transform.get_phase_map_at_theta(theta_idx)
        H_volt, u_volt = transform.get_volt_map_at_theta(theta_idx, omega)
        curr_max = transform.machine.curr_max
        volt_max = transform.machine.volt_max

        constraints: list[dict[str, Any]] = []

        for p in range(H_ph.shape[0]):
            hp = H_ph[p].copy()
            constraints.append({"type": "ineq", "fun": lambda x, hp=hp: curr_max - float(hp @ x)})
            constraints.append({"type": "ineq", "fun": lambda x, hp=hp: curr_max + float(hp @ x)})

        for p in range(H_volt.shape[0]):
            hv = H_volt[p].copy()
            uv = float(u_volt[p])
            constraints.append({"type": "ineq", "fun": lambda x, hv=hv, uv=uv: volt_max - (float(hv @ x) + uv)})
            constraints.append({"type": "ineq", "fun": lambda x, hv=hv, uv=uv: volt_max + (float(hv @ x) + uv)})

        constraints.extend(transform.get_extra_constraints_at_theta(theta_idx))

        return constraints

    def _run_optimization(
        self,
        objective_fun,
        constraints: list[dict[str, Any]],
        candidates: list[np.ndarray],
        opts: dict[str, Any],
    ) -> tuple[np.ndarray, float, bool]:
        best_res = None
        best_val = float("inf")
        for vec_start in candidates:
            res = minimize(objective_fun, vec_start, method="SLSQP", constraints=constraints, options=opts)
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_res = res
        if best_res is not None:
            return best_res.x, best_res.fun, True
        return np.full(candidates[0].shape, np.nan), float("inf"), False

    def maximize_torque_at_theta(
        self,
        theta_idx: int,
        omega: float,
        transform: BaseTransform,
        curr_dq_guess: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float, bool]:
        """
        Maximises torque at a single rotor angle subject to instantaneous
        phase-current and phase-voltage limits at that angle, plus the
        transform's extra constraints (null-space equality for two-fault
        modes — pulled in automatically via
        ``_build_constraints_at_theta``).

        Returns:
            (optimal_curr_dq, max_torque, success)
        """
        constraints = self._build_constraints_at_theta(theta_idx, omega, transform)
        candidates = self._candidates(curr_dq_guess)

        def objective(curr_dq: np.ndarray) -> float:
            return -self.model.calculate_torque(omega, curr_dq)

        curr_dq, neg_torq, success = self._run_optimization(objective, constraints, candidates, self.opts)
        return curr_dq, -neg_torq, success

    def compute_torque_envelope(
        self,
        omega: float,
        transform: BaseTransform,
    ) -> np.ndarray:
        """
        Computes the per-θ maximum-torque envelope.

        Returns:
            np.ndarray of shape (n_theta,): T_max at each sampled rotor angle.
            Failed solves are recorded as NaN.
        """
        theta_indices = self._get_theta_indices(transform)
        envelope = np.full(len(theta_indices), np.nan)
        curr_dq_guess: np.ndarray | None = None

        for k, theta_idx in enumerate(theta_indices):
            curr_dq, torq, success = self.maximize_torque_at_theta(
                theta_idx, omega, transform, curr_dq_guess
            )
            if success:
                envelope[k] = torq
                curr_dq_guess = curr_dq

        return envelope

    def get_rated_torque(
        self,
        omega: float,
        transform: BaseTransform,
    ) -> tuple[float, bool]:
        """
        Returns ``(min_theta T_max(theta), success)``.

        The minimum of the torque envelope is the largest torque that can be
        sustained continuously across all rotor positions. Any θ where the
        SLSQP solve fails is treated as infeasible — the whole operating
        point is reported as infeasible rather than skipping that θ in the
        min, because the paper-spec rated torque must hold at every rotor
        angle.
        """
        envelope = self.compute_torque_envelope(omega, transform)
        if np.any(np.isnan(envelope)):
            return 0.0, False
        T_min = float(np.min(envelope))
        if T_min <= 0.0:
            return 0.0, False
        return T_min, True

    def minimize_current_at_theta(
        self,
        theta_idx: int,
        torq_target: float,
        omega: float,
        transform: BaseTransform,
        curr_dq_guess: np.ndarray | None = None,
    ) -> tuple[np.ndarray, bool]:
        """
        Finds the minimum-norm DQ current vector that produces ``torq_target``
        at the given rotor angle, subject to instantaneous phase limits at
        that angle only.

        Returns:
            (optimal_curr_dq, success)
        """
        constraints = self._build_constraints_at_theta(theta_idx, omega, transform)

        def eq_cons(curr_dq: np.ndarray) -> float:
            return self.model.calculate_torque(omega, curr_dq) - torq_target

        constraints.append({"type": "eq", "fun": eq_cons})
        candidates = self._candidates(curr_dq_guess)

        def objective(curr_dq: np.ndarray) -> float:
            return float(np.sum(curr_dq**2))

        curr_dq, _, success = self._run_optimization(objective, constraints, candidates, self.opts)
        return curr_dq, success
