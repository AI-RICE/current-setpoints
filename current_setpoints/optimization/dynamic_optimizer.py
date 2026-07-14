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

    def _precompute_maps(self, omega: float) -> dict[str, Any]:
        fwd = self.fwd
        theta_grid, idx_grid = self._grid_indices()
        kept = list(fwd.fault._kept)

        H_curr = np.stack([fwd.phase_map_at_theta(int(n)) for n in idx_grid])
        H_curr = H_curr.transpose(1, 0, 2)  # (n_surv, n_grid, dim)

        gU_list, gL_list, bV_list = [], [], []
        for n in idx_grid:
            gU, gL, bV = fwd.volt_map_at_theta(int(n), omega)
            gU_list.append(gU[kept])
            gL_list.append(gL[kept])
            bV_list.append(bV[kept])

        return {
            "theta_grid": theta_grid,
            "idx_grid": idx_grid,
            "H_curr": H_curr,  # (n_surv, n_grid, dim) current map
            "G_volt": np.stack(gU_list).transpose(1, 0, 2),  # (n_surv, n_grid, dim) static voltage
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
        H_curr = maps["H_curr"]  # (n_surv, n_grid, dim)
        G_volt = maps["G_volt"]  # (n_surv, n_grid, dim)
        B = maps["bemf_ph"]  # (n_surv, n_grid)
        n_surv = H_curr.shape[0]
        curr_max = fwd.drive.curr_max
        volt_max = fwd.drive.volt_max

        cons: list[dict[str, Any]] = []
        for k in range(n_surv):
            h = H_curr[k, n].copy()
            cons.append({"type": "ineq", "fun": lambda x, h=h: curr_max - h @ x})
            cons.append({"type": "ineq", "fun": lambda x, h=h: curr_max + h @ x})
            g = G_volt[k, n].copy()
            b = float(B[k, n])
            cons.append({"type": "ineq", "fun": lambda x, g=g, b=b: volt_max - (g @ x + b)})
            cons.append({"type": "ineq", "fun": lambda x, g=g, b=b: volt_max + (g @ x + b)})

        if torq_target is not None:
            T = torq_target
            cons.append({"type": "eq", "fun": lambda x, T=T: fwd.drive.torque(omega, x) - T})

        cons.extend(fwd.extra_constraints_at_theta(n))
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
