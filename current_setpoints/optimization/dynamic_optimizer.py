from __future__ import annotations

from typing import Any

import numpy as np

from ..models.forward_model import ForwardModel
from .optimizer import BaseOptimizer, Solution, _run_slsqp


class DynamicOptimizer(BaseOptimizer):
    def __init__(
        self,
        fwd: ForwardModel,
        opts: dict[str, Any] | None = None,
        n_grid: int = 64,
    ) -> None:
        super().__init__(fwd, opts)
        self.n_grid = n_grid

    def _grid_indices(self) -> tuple[np.ndarray, np.ndarray]:
        n_theta = self.fwd.vec_theta.size - 1
        theta_grid = np.linspace(0.0, 2 * np.pi, self.n_grid, endpoint=False)
        idx_grid = np.round(theta_grid / (2 * np.pi) * n_theta).astype(int) % n_theta
        return theta_grid, idx_grid

    def _precompute_maps(self, omega: float, lin_curr: np.ndarray | None = None) -> dict[str, Any]:
        """lin_curr: the current to linearize the voltage operator/inductance
        around (drive.voltage_operator/inductance/bemf_dq are evaluated at
        this point, not the actual per-node candidate) -- either a single
        (dim,) vector applied at every node, or a (n_grid, dim) array giving
        each node its own linearization point (see ActiveSetOptimizer, which
        re-linearizes around its R0 solution). Exact for a linear machine
        (ConstantFlux -- these don't depend on current at all, so any
        linearization point gives the same, exact answer) but only an
        approximation for a genuinely current-dependent flux model
        (NeuralFlux) -- callers that can re-linearize around a realistic
        operating current should pass one instead of relying on the zeros
        default. Only feeds ActiveSetOptimizer's joint solve now --
        IndependentOptimizer's own per-node constraints evaluate voltage
        exactly at the actual candidate, not through this linearization."""
        fwd = self.fwd
        theta_grid, idx_grid = self._grid_indices()
        kept = list(fwd.fault._kept)
        if lin_curr is None:
            lin_curr = np.zeros(fwd.drive.dim)
        lin_curr = np.asarray(lin_curr)
        per_node = lin_curr.ndim == 2

        H_curr = np.stack([fwd.phase_map_at_theta(int(n)) for n in idx_grid])
        H_curr = H_curr.transpose(1, 0, 2)  # (n_surv, n_grid, dim)
        H_volt = np.stack([fwd._mat_dq_to_ph_all[int(n)][kept] for n in idx_grid])
        H_volt = H_volt.transpose(1, 0, 2)  # (n_surv, n_grid, dim) -- geometric only, no U/L/bemf baked in

        gU_list, gL_list, bV_list = [], [], []
        for idx, n in enumerate(idx_grid):
            node_curr = lin_curr[idx] if per_node else lin_curr
            gU, gL, bV = fwd.volt_map_at_theta(int(n), omega, node_curr)
            gU_list.append(gU[kept])
            gL_list.append(gL[kept])
            bV_list.append(bV[kept])

        return {
            "theta_grid": theta_grid,
            "idx_grid": idx_grid,
            "H_curr": H_curr,  # (n_surv, n_grid, dim) current map
            "H_volt": H_volt,  # (n_surv, n_grid, dim) geometric voltage map (current-independent)
            "G_volt": np.stack(gU_list).transpose(1, 0, 2),  # (n_surv, n_grid, dim) linearized voltage
            "L_proj": np.stack(gL_list).transpose(1, 0, 2),  # (n_surv, n_grid, dim) inductive
            "bemf_ph": np.stack(bV_list).T,  # (n_surv, n_grid) BEMF offset
        }


class IndependentOptimizer(DynamicOptimizer):
    def _per_angle_constraints(
        self,
        maps: dict[str, Any],
        omega: float,
        n: int,
        torq_target: float | None,
    ) -> list[dict[str, Any]]:
        fwd = self.fwd
        H_curr_n = maps["H_curr"][:, n, :]  # (n_surv, dim)
        H_volt_n = maps["H_volt"][:, n, :]  # (n_surv, dim) -- geometric only, no U/L/bemf baked in
        fine_idx = int(maps["idx_grid"][n])  # extra_constraints_at_theta indexes fwd.vec_theta, not the coarse n_grid
        curr_max = fwd.drive.curr_max
        volt_max = fwd.drive.volt_max

        def curr_con(x: np.ndarray) -> np.ndarray:
            return curr_max - np.abs(H_curr_n @ x)

        def volt_con(x: np.ndarray) -> np.ndarray:
            # Exact for any drive: voltage_operator/bemf_dq evaluated at the
            # actual candidate x, not a fixed linearization point -- unlike
            # ActiveSetOptimizer's joint solve, each node here is independent
            # (no inter-node coupling), so nothing requires this to stay linear.
            v_dq = fwd.drive.voltage_operator(omega, x) @ x + fwd.drive.bemf_dq(omega, x)
            return volt_max - np.abs(H_volt_n @ v_dq)

        cons: list[dict[str, Any]] = [
            {"type": "ineq", "fun": curr_con},
            {"type": "ineq", "fun": volt_con},
        ]

        if torq_target is not None:
            T = torq_target
            cons.append({"type": "eq", "fun": lambda x, T=T: fwd.drive.torque(omega, x) - T})

        cons.extend(fwd.extra_constraints_at_theta(fine_idx))
        return cons

    def _run_per_angle(
        self,
        omega: float,
        maps: dict[str, Any],
        torq_target: float | None,
        maximize: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        fwd = self.fwd
        n_grid = self.n_grid
        dim = fwd.drive.dim

        curr_dq_grid = np.full((n_grid, dim), np.nan)
        ok_grid = np.zeros(n_grid, dtype=bool)
        seed = fwd.drive.seeds()[0]

        for i in range(n_grid):
            cons = self._per_angle_constraints(maps, omega, i, None if maximize else torq_target)
            obj = (lambda x: -fwd.drive.torque(omega, x)) if maximize else (lambda x: float(np.sum(x**2)))
            best_x, _, ok = _run_slsqp(obj, cons, fwd.drive.seeds(seed), self.opts)
            curr_dq_grid[i] = best_x
            ok_grid[i] = ok
            if ok:
                seed = best_x

        return curr_dq_grid, ok_grid

    def minimize_current(
        self,
        torq_target: float,
        omega: float,
        guess: np.ndarray | None = None,
    ) -> Solution:
        maps = self._precompute_maps(omega)
        curr_dq_grid, ok_grid = self._run_per_angle(omega, maps, torq_target)
        success = bool(np.all(ok_grid))
        mean_dq = np.nanmean(curr_dq_grid, axis=0)
        torque = float(self.fwd.drive.torque(omega, mean_dq)) if np.any(ok_grid) else float("nan")
        return Solution(
            curr_dq=curr_dq_grid,
            torque=torque,
            success=success,
            diagnostics={"ok_grid": ok_grid, "theta_grid": maps["theta_grid"]},
        )

    def maximize_torque(
        self,
        omega: float,
        guess: np.ndarray | None = None,
    ) -> Solution:
        maps = self._precompute_maps(omega)
        curr_dq_grid, ok_grid = self._run_per_angle(omega, maps, None, maximize=True)
        if not np.any(ok_grid):
            return Solution(
                curr_dq=np.full((self.n_grid, self.fwd.drive.dim), np.nan),
                torque=float("nan"),
                success=False,
            )
        torques = np.array(
            [self.fwd.drive.torque(omega, curr_dq_grid[i]) if ok_grid[i] else np.nan for i in range(self.n_grid)]
        )
        # Most constrained angle sets the feasible max torque for the whole trajectory
        t_max = float(np.nanmin(torques))
        return Solution(
            curr_dq=curr_dq_grid,
            torque=t_max,
            success=bool(np.all(ok_grid)),
            diagnostics={"ok_grid": ok_grid, "theta_grid": maps["theta_grid"]},
        )
