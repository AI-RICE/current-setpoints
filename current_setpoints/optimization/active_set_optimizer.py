from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.optimize import minimize

from ..models.forward_model import ForwardModel
from .dynamic_optimizer import IndependentOptimizer
from .optimizer import Solution, _constraints_satisfied


class ActiveSetOptimizer(IndependentOptimizer):
    def __init__(
        self,
        fwd: ForwardModel,
        opts: dict[str, Any] | None = None,
        n_grid: int = 32,
        scheme: str = "forward",
        max_outer_iter: int | None = None,
        tol: float = 1e-3,
        rho: float = 0.0,
        iron_weight: float = 0.0,
        excess_weight: float = 0.0,
        torque_ripple: float = 0.0,
        torque_ripple_relative: bool = False,
        slew_limit: float | Sequence[float | None] | None = None,
        legacy_algorithm1: bool = False,
    ) -> None:
        super().__init__(
            fwd,
            opts if opts is not None else {"disp": False, "ftol": 1e-7, "maxiter": 1500, "eps": 1e-8},
            n_grid,
        )
        if scheme not in ("forward", "central"):
            raise ValueError(f"scheme must be 'forward' or 'central', got {scheme!r}")
        self.scheme = scheme
        self.max_outer_iter = max_outer_iter if max_outer_iter is not None else n_grid
        self.tol = tol
        self.rho = rho
        self.iron_weight = iron_weight
        self.excess_weight = excess_weight
        if torque_ripple < 0.0:
            raise ValueError(f"torque_ripple must be >= 0, got {torque_ripple!r}")
        self.torque_ripple = torque_ripple
        self.torque_ripple_relative = torque_ripple_relative
        if slew_limit is not None:
            limits = [slew_limit] if np.isscalar(slew_limit) else list(slew_limit)
            if not np.isscalar(slew_limit) and len(limits) != fwd.drive.dim // 2:
                raise ValueError(f"slew_limit needs one value per plane ({fwd.drive.dim // 2}), got {len(limits)}")
            if any(s is not None and s <= 0.0 for s in limits):
                raise ValueError(f"slew limits must be > 0, got {slew_limit!r}")
        self.slew_limit = slew_limit
        self.legacy_algorithm1 = legacy_algorithm1

    def _ripple_band(self, torq_target: float) -> float:
        return self.torque_ripple * abs(torq_target) if self.torque_ripple_relative else self.torque_ripple

    _SLEW_TOL = 1e-3

    def _slew_steps(self, omega: float) -> list[tuple[slice, float]]:
        if self.slew_limit is None or omega == 0.0:
            return []
        dt_ms = 1000.0 * (2 * np.pi / self.n_grid) / abs(omega)
        if np.isscalar(self.slew_limit):
            return [(slice(None), float(self.slew_limit) * dt_ms)]
        return [(slice(2 * k, 2 * k + 2), float(s) * dt_ms) for k, s in enumerate(self.slew_limit) if s is not None]

    def _slew_excess(self, X: np.ndarray, omega: float) -> float:
        d = np.roll(X, -1, axis=0) - X
        return max((float(np.max(np.linalg.norm(d[:, sl], axis=1))) / step - 1.0 for sl, step in self._slew_steps(omega)),
                   default=0.0)

    def voltage_residuals(
        self,
        maps: dict[str, Any],
        omega: float,
        X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_grid, _dim = X.shape
        delta_theta = 2 * np.pi / n_grid
        G = maps["G_volt"]
        Lp = maps["L_proj"]
        B = maps["bemf_ph"]

        if self.scheme == "forward":
            dxdt = (np.roll(X, -1, axis=0) - X) / delta_theta
        else:
            dxdt = (np.roll(X, -1, axis=0) - np.roll(X, 1, axis=0)) / (2.0 * delta_theta)

        v_ph = np.einsum("knj,nj->kn", G, X) + omega * np.einsum("knj,nj->kn", Lp, dxdt) + B
        v_abs = np.max(np.abs(v_ph), axis=0)
        if self.scheme == "forward":
            v_end = np.roll(np.einsum("knj,nj->kn", G, X) + B, -1, axis=1) + omega * np.einsum(
                "knj,nj->kn", np.roll(Lp, -1, axis=1), dxdt
            )
            v_abs = np.maximum(v_abs, np.max(np.abs(v_end), axis=0))
        r = v_abs - self.fwd.drive.volt_max
        return r, v_ph.T

    def _solve_joint(
        self,
        maps: dict[str, Any],
        omega: float,
        torq_target: float,
        X_init: np.ndarray,
        active_nodes: set[int],
    ) -> tuple[np.ndarray, bool]:
        fwd = self.fwd
        n_grid, dim = X_init.shape
        n_surv = len(fwd.fault._kept)
        delta_theta = 2 * np.pi / n_grid
        curr_max = fwd.drive.curr_max
        volt_max = fwd.drive.volt_max
        total_vars = n_grid * dim
        coupling = omega / delta_theta
        coupling_c = omega / (2.0 * delta_theta)

        obj_scale = 1.0 / (n_grid * curr_max**2)
        torq_scale = max(abs(torq_target), 1.0)

        H_curr = maps["H_curr"]
        G_volt = maps["G_volt"]
        Lp = maps["L_proj"]
        B = maps["bemf_ph"]

        def slot(n: int) -> slice:
            return slice(n * dim, (n + 1) * dim)

        iron_active = self.iron_weight > 0.0 or self.excess_weight > 0.0
        iron_scale = (omega**2) / (delta_theta**2) / n_grid
        excess_scale = (omega / delta_theta) ** 1.5 / n_grid
        iron_bias = np.roll(B, -1, axis=1) - B

        def _iron_term(X_flat: np.ndarray) -> tuple[float, np.ndarray]:
            if not iron_active:
                return 0.0, np.zeros_like(X_flat)
            X = X_flat.reshape(n_grid, dim)
            proj = np.einsum("knj,nj->kn", Lp, X)
            delta = np.roll(proj, -1, axis=1) - proj + iron_bias
            val = 0.0
            w = np.zeros_like(delta)
            if self.iron_weight > 0.0:
                val += self.iron_weight * iron_scale * float(np.sum(delta**2))
                w += 2.0 * self.iron_weight * iron_scale * delta
            if self.excess_weight > 0.0:
                ad = np.abs(delta)
                val += self.excess_weight * excess_scale * float(np.sum(ad**1.5))
                w += 1.5 * self.excess_weight * excess_scale * np.sign(delta) * np.sqrt(ad)
            grad_X = np.einsum("knj,kn->nj", Lp, np.roll(w, 1, axis=1) - w)
            return val, grad_X.flatten()

        rho = self.rho
        if rho > 0.0:
            slew_pairs = [(n, (n + 1) % n_grid) for n in range(n_grid)]

            def objective(X_flat: np.ndarray) -> float:
                slew = sum(float(np.dot(d := X_flat[slot(a)] - X_flat[slot(b)], d)) for a, b in slew_pairs)
                iv, _ = _iron_term(X_flat)
                return float(np.sum(X_flat**2)) + rho * slew + iv

            def objective_grad(X_flat: np.ndarray) -> np.ndarray:
                g = 2.0 * X_flat.copy()
                for a, b in slew_pairs:
                    d = X_flat[slot(a)] - X_flat[slot(b)]
                    g[slot(a)] += 2.0 * rho * d
                    g[slot(b)] -= 2.0 * rho * d
                _, ig = _iron_term(X_flat)
                return g + ig
        else:

            def objective(X_flat: np.ndarray) -> float:
                iv, _ = _iron_term(X_flat)
                return float(np.sum(X_flat**2)) + iv

            def objective_grad(X_flat: np.ndarray) -> np.ndarray:
                _, ig = _iron_term(X_flat)
                return 2.0 * X_flat + ig

        constraints: list[dict[str, Any]] = []

        torque_grad = getattr(fwd.drive, "torque_grad", None)
        ripple = self._ripple_band(torq_target)
        for n in range(n_grid) if ripple == 0.0 else ():

            def eq_torq(X_flat: np.ndarray, _n: int = n) -> float:
                return (fwd.drive.torque(omega, X_flat[slot(_n)]) - torq_target) / torq_scale

            con: dict[str, Any] = {"type": "eq", "fun": eq_torq}
            if torque_grad is not None:

                def eq_torq_jac(X_flat: np.ndarray, _n: int = n) -> np.ndarray:
                    g = np.zeros(total_vars)
                    g[slot(_n)] = torque_grad(omega, X_flat[slot(_n)]) / torq_scale
                    return g

                con["jac"] = eq_torq_jac
            constraints.append(con)

        if ripple > 0.0:
            def mean_torq(X_flat: np.ndarray) -> float:
                t = [fwd.drive.torque(omega, X_flat[slot(n)]) for n in range(n_grid)]
                return (float(np.mean(t)) - torq_target) / torq_scale

            con = {"type": "eq", "fun": mean_torq}
            if torque_grad is not None:

                def mean_torq_jac(X_flat: np.ndarray) -> np.ndarray:
                    g = np.zeros(total_vars)
                    for n in range(n_grid):
                        g[slot(n)] = torque_grad(omega, X_flat[slot(n)]) / (n_grid * torq_scale)
                    return g

                con["jac"] = mean_torq_jac
            constraints.append(con)

            for n in range(n_grid):
                for sign in (1.0, -1.0):

                    def band_torq(X_flat: np.ndarray, _n: int = n, _s: float = sign) -> float:
                        return (ripple - _s * (fwd.drive.torque(omega, X_flat[slot(_n)]) - torq_target)) / torq_scale

                    con = {"type": "ineq", "fun": band_torq}
                    if torque_grad is not None:

                        def band_torq_jac(X_flat: np.ndarray, _n: int = n, _s: float = sign) -> np.ndarray:
                            g = np.zeros(total_vars)
                            g[slot(_n)] = -_s * torque_grad(omega, X_flat[slot(_n)]) / torq_scale
                            return g

                        con["jac"] = band_torq_jac
                    constraints.append(con)

        def _lin(coef: np.ndarray, offset: float, scale: float) -> dict[str, Any]:
            coef_s, offset_s = coef / scale, offset / scale

            def fun(X_flat: np.ndarray) -> float:
                return float(coef_s @ X_flat + offset_s)

            def jac(X_flat: np.ndarray) -> np.ndarray:
                return coef_s

            return {"type": "ineq", "fun": fun, "jac": jac}

        for n in range(n_grid):
            for k in range(n_surv):
                h = H_curr[k, n]
                c_pos = np.zeros(total_vars)
                c_pos[slot(n)] = -h
                c_neg = np.zeros(total_vars)
                c_neg[slot(n)] = h
                constraints.append(_lin(c_pos, curr_max, curr_max))
                constraints.append(_lin(c_neg, curr_max, curr_max))

        for n in range(n_grid):
            n_next = (n + 1) % n_grid
            n_prev = (n - 1) % n_grid
            for k in range(n_surv):
                g_n = G_volt[k, n]
                lp_n = Lp[k, n]
                b_n = float(B[k, n])

                if n not in active_nodes and not self.legacy_algorithm1:
                    continue
                if n not in active_nodes:
                    c_pos = np.zeros(total_vars)
                    c_pos[slot(n)] = -g_n
                    c_neg = np.zeros(total_vars)
                    c_neg[slot(n)] = g_n
                    constraints.append(_lin(c_pos, volt_max - b_n, volt_max))
                    constraints.append(_lin(c_neg, volt_max + b_n, volt_max))

                elif self.scheme == "forward":
                    g_nn = G_volt[k, n_next]
                    lp_nn = Lp[k, n_next]
                    b_nn = float(B[k, n_next])
                    gm = g_n - coupling * lp_n
                    gp = coupling * lp_n
                    for sign in (1.0, -1.0):
                        c = np.zeros(total_vars)
                        c[slot(n)] = sign * (-gm)
                        c[slot(n_next)] = sign * (-gp)
                        constraints.append(_lin(c, volt_max - sign * b_n, volt_max))
                    gmr = -coupling * lp_nn
                    gpr = g_nn + coupling * lp_nn
                    for sign in (1.0, -1.0):
                        c = np.zeros(total_vars)
                        c[slot(n)] = sign * (-gmr)
                        c[slot(n_next)] = sign * (-gpr)
                        constraints.append(_lin(c, volt_max - sign * b_nn, volt_max))

                else:
                    a_self = g_n
                    a_next = coupling_c * lp_n
                    a_prev = -coupling_c * lp_n
                    for sign in (1.0, -1.0):
                        c = np.zeros(total_vars)
                        c[slot(n)] = sign * (-a_self)
                        c[slot(n_next)] = sign * (-a_next)
                        c[slot(n_prev)] = sign * (-a_prev)
                        constraints.append(_lin(c, volt_max - sign * b_n, volt_max))

        idx_grid = maps["idx_grid"]
        for n in range(n_grid):
            for fc in fwd.extra_constraints_at_theta(int(idx_grid[n])):
                _n = n
                _f = fc["fun"]

                def lifted(X_flat: np.ndarray, _n: int = _n, _f: Any = _f) -> float:
                    return _f(X_flat[slot(_n)])

                constraints.append({"type": "eq", "fun": lifted})

        for sl, step in self._slew_steps(omega):

            def slew_con(X_flat: np.ndarray, sl: slice = sl, step: float = step) -> np.ndarray:
                X = X_flat.reshape(n_grid, dim)
                d = (np.roll(X, -1, axis=0) - X)[:, sl]
                return (step**2 - np.sum(d * d, axis=1)) / step**2

            def slew_jac(X_flat: np.ndarray, sl: slice = sl, step: float = step) -> np.ndarray:
                X = X_flat.reshape(n_grid, dim)
                d = np.zeros_like(X)
                d[:, sl] = (np.roll(X, -1, axis=0) - X)[:, sl]
                J = np.zeros((n_grid, total_vars))
                rows = np.arange(n_grid)
                for j in range(dim):
                    J[rows, rows * dim + j] += 2.0 * d[:, j] / step**2
                    J[rows, ((rows + 1) % n_grid) * dim + j] -= 2.0 * d[:, j] / step**2
                return J

            constraints.append({"type": "ineq", "fun": slew_con, "jac": slew_jac})

        res = minimize(
            lambda X_flat: obj_scale * objective(X_flat),
            X_init.flatten(),
            jac=lambda X_flat: obj_scale * objective_grad(X_flat),
            method="SLSQP",
            constraints=constraints,
            options=self.opts,
        )
        return res.x.reshape(n_grid, dim), bool(res.success)

    def _nodes_violating_static(
        self,
        maps: dict[str, Any],
        omega: float,
        torq_target: float,
        X: np.ndarray,
    ) -> list[int]:
        fwd = self.fwd
        curr_max = fwd.drive.curr_max
        ripple = self._ripple_band(torq_target)
        if ripple == 0.0:
            torq_con: dict[str, Any] = {"type": "eq", "fun": lambda x: fwd.drive.torque(omega, x) - torq_target}
        else:
            torq_con = {"type": "ineq", "fun": lambda x: ripple - abs(fwd.drive.torque(omega, x) - torq_target)}
        bad = []
        for n in range(X.shape[0]):
            H_curr_n = maps["H_curr"][:, n, :]
            cons: list[dict[str, Any]] = [
                torq_con,
                {"type": "ineq", "fun": lambda x, H=H_curr_n: curr_max - np.abs(H @ x)},
            ]
            cons.extend(fwd.extra_constraints_at_theta(int(maps["idx_grid"][n])))
            if not _constraints_satisfied(cons, X[n]):
                bad.append(n)
        return bad

    def minimize_current(
        self,
        torq_target: float,
        omega: float,
        guess: np.ndarray | None = None,
    ) -> Solution:
        maps = self._precompute_maps(omega)

        r0_grid, ok0 = self._run_per_angle(omega, maps, torq_target, guess=guess)
        r0_free_nodes: list[int] = []
        if not bool(np.all(ok0)) and not self.legacy_algorithm1:
            free_grid, ok_free = self._run_per_angle(omega, maps, torq_target, guess=guess, voltage=False)
            for n in np.where(~ok0)[0]:
                if ok_free[n]:
                    r0_grid[n], ok0[n] = free_grid[n], True
                    r0_free_nodes.append(int(n))
        if not bool(np.all(ok0)):
            return Solution(
                curr_dq=r0_grid,
                torque=float("nan"),
                success=False,
                diagnostics={
                    "r0_ok": ok0,
                    "theta_grid": maps["theta_grid"],
                    "fail_reason": "R0 failed at some nodes" if self.legacy_algorithm1
                    else "torque/current/open-phase constraints infeasible at some nodes",
                },
            )

        maps = self._precompute_maps(omega, lin_curr=r0_grid)

        X = r0_grid.copy()
        active: frozenset[int] = frozenset()
        history: list[dict[str, Any]] = []
        converged = False
        fail_reason: str | None = None
        joint_done = False

        for outer in range(self.max_outer_iter):
            r, _ = self.voltage_residuals(maps, omega, X)
            violating = frozenset(int(n) for n in np.where(r > self.tol)[0])
            entry: dict[str, Any] = {
                "iter": outer,
                "max_V_residual": float(np.max(r)),
                "n_violating": len(violating),
                "active_size": len(active),
            }

            if not violating:
                bad_nodes = self._nodes_violating_static(maps, omega, torq_target, X)
                mean_err = abs(float(np.mean(self._node_torques(omega, X))) - torq_target)
                slew_excess = self._slew_excess(X, omega)
                if bad_nodes:
                    history.append(entry | {"action": "voltage ok, other constraints violated", "bad_nodes": bad_nodes})
                    fail_reason = f"torque/current/fault constraint violated at {len(bad_nodes)} nodes"
                    break
                if self._ripple_band(torq_target) > 0.0 and mean_err > 1e-4:
                    history.append(entry | {"action": "voltage ok, mean torque off target", "mean_err": mean_err})
                    fail_reason = f"mean torque off target by {mean_err:.2e} Nm"
                    break
                if slew_excess <= self._SLEW_TOL:
                    history.append(entry | {"action": "converged"})
                    converged = True
                    break
                if joint_done:
                    history.append(entry | {"action": "voltage ok, slew limit violated", "slew_excess": slew_excess})
                    fail_reason = f"slew limit exceeded by {100 * slew_excess:.2f} %"
                    break
                entry["slew_excess"] = slew_excess

            new_active = active | violating

            X_new, ok = self._solve_joint(maps, omega, torq_target, X, set(new_active))
            entry["action"] = f"solve(|A|={len(new_active)})"
            entry["solve_success"] = ok
            history.append(entry)
            if not ok:
                fail_reason = f"joint solve failed at outer iteration {outer}"
                break
            X = X_new
            active = new_active
            joint_done = True
            maps = self._precompute_maps(omega, lin_curr=X)
        else:
            fail_reason = f"max_outer_iter={self.max_outer_iter} reached"

        node_torques = self._node_torques(omega, X)
        return Solution(
            curr_dq=X,
            torque=float(np.nanmean(node_torques)),
            success=converged,
            diagnostics={
                "active": active,
                "history": history,
                "r0_grid": r0_grid,
                "r0_voltage_free_nodes": r0_free_nodes,
                "theta_grid": maps["theta_grid"],
                "torque_min": float(np.nanmin(node_torques)),
                "torque_max": float(np.nanmax(node_torques)),
                "fail_reason": fail_reason,
            },
        )

    _N_BISECT_ITERS = 9

    def maximize_torque(
        self,
        omega: float,
        guess: np.ndarray | None = None,
        start_margin: float = 1.0,
    ) -> Solution:
        maps = self._precompute_maps(omega)
        r0_grid, ok0 = self._run_per_angle(omega, maps, None, maximize=True, guess=guess,
                                           voltage=self.legacy_algorithm1)

        if not np.any(ok0):
            return Solution(
                curr_dq=np.full((self.n_grid, self.fwd.drive.dim), np.nan),
                torque=float("nan"),
                success=False,
            )

        torques = np.array([self.fwd.drive.torque(omega, r0_grid[i]) if ok0[i] else np.nan for i in range(self.n_grid)])
        t_naive = float(np.nanmin(torques))
        n_bisect = self._N_BISECT_ITERS
        if not self.legacy_algorithm1:
            if self.torque_ripple_relative and 0.0 < self.torque_ripple < 1.0:
                t_naive = t_naive / (1.0 - self.torque_ripple)
            elif not self.torque_ripple_relative:
                t_naive = t_naive + self.torque_ripple
            n_bisect += 3

        lo, hi = 0.0, t_naive
        best_sol: Solution | None = None
        best_margin = 0.0

        sol = self.minimize_current(start_margin * t_naive, omega, guess=guess)
        if sol.success:
            lo = start_margin * t_naive
            best_sol, best_margin = sol, start_margin

        for _ in range(n_bisect):
            if lo >= hi:
                break
            mid = 0.5 * (lo + hi)
            sol = self.minimize_current(mid, omega, guess=guess)
            if sol.success:
                lo = mid
                best_sol, best_margin = sol, (mid / t_naive if t_naive > 0.0 else 0.0)
            else:
                hi = mid

        if best_sol is not None:
            best_sol.diagnostics["margin"] = best_margin
            return best_sol
        return sol
