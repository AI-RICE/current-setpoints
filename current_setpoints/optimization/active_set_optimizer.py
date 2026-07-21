from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize

from ..models.forward_model import ForwardModel
from .dynamic_optimizer import IndependentOptimizer
from .optimizer import Solution


class ActiveSetOptimizer(IndependentOptimizer):
    def __init__(
        self,
        fwd: ForwardModel,
        opts: dict[str, Any] | None = None,
        n_grid: int = 32,
        scheme: str = "forward",
        max_outer_iter: int = 8,
        tol: float = 1e-3,
        rho: float = 0.0,
        iron_weight: float = 0.0,
        excess_weight: float = 0.0,
    ) -> None:
        super().__init__(
            fwd,
            opts if opts is not None else {"disp": False, "ftol": 1e-7, "maxiter": 1500, "eps": 1e-8},
            n_grid,
        )
        if scheme not in ("forward", "central"):
            raise ValueError(f"scheme must be 'forward' or 'central', got {scheme!r}")
        self.scheme = scheme
        self.max_outer_iter = max_outer_iter
        self.tol = tol
        self.rho = rho
        self.iron_weight = iron_weight
        self.excess_weight = excess_weight

    def voltage_residuals(
        self,
        maps: dict[str, Any],
        omega: float,
        X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_grid, _dim = X.shape
        delta_theta = 2 * np.pi / n_grid
        G = maps["G_volt"]  # (n_surv, n_grid, dim)
        Lp = maps["L_proj"]  # (n_surv, n_grid, dim)
        B = maps["bemf_ph"]  # (n_surv, n_grid)

        if self.scheme == "forward":
            dxdt = (np.roll(X, -1, axis=0) - X) / delta_theta
        else:
            dxdt = (np.roll(X, -1, axis=0) - np.roll(X, 1, axis=0)) / (2.0 * delta_theta)

        v_ph = np.einsum("knj,nj->kn", G, X) + omega * np.einsum("knj,nj->kn", Lp, dxdt) + B  # (n_surv, n_grid)
        r = np.max(np.abs(v_ph), axis=0) - self.fwd.drive.volt_max
        return r, v_ph.T  # (n_grid,), (n_grid, n_surv)

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

        H_curr = maps["H_curr"]  # (n_surv, n_grid, dim)
        G_volt = maps["G_volt"]  # (n_surv, n_grid, dim)
        Lp = maps["L_proj"]  # (n_surv, n_grid, dim)
        B = maps["bemf_ph"]  # (n_surv, n_grid)

        def slot(n: int) -> slice:
            return slice(n * dim, (n + 1) * dim)

        iron_active = self.iron_weight > 0.0 or self.excess_weight > 0.0
        iron_scale = (omega**2) / (delta_theta**2) / n_grid
        excess_scale = (omega / delta_theta) ** 1.5 / n_grid
        # PM flux change proxy between adjacent nodes (shape n_surv × n_grid)
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

        # Torque equalities (nonlinear; SLSQP uses finite-diff Jacobian)
        for n in range(n_grid):

            def eq_torq(X_flat: np.ndarray, _n: int = n) -> float:
                return fwd.drive.torque(omega, X_flat[slot(_n)]) - torq_target

            constraints.append({"type": "eq", "fun": eq_torq})

        def _lin(coef: np.ndarray, offset: float) -> dict[str, Any]:
            def fun(X_flat: np.ndarray) -> float:
                return float(coef @ X_flat + offset)

            def jac(X_flat: np.ndarray) -> np.ndarray:
                return coef

            return {"type": "ineq", "fun": fun, "jac": jac}

        # Current: always static (no inductive coupling)
        for n in range(n_grid):
            for k in range(n_surv):
                h = H_curr[k, n]
                c_pos = np.zeros(total_vars)
                c_pos[slot(n)] = -h
                c_neg = np.zeros(total_vars)
                c_neg[slot(n)] = h
                constraints.append(_lin(c_pos, curr_max))
                constraints.append(_lin(c_neg, curr_max))

        # Voltage: static outside active set, dynamic inside
        for n in range(n_grid):
            n_next = (n + 1) % n_grid
            n_prev = (n - 1) % n_grid
            for k in range(n_surv):
                g_n = G_volt[k, n]
                lp_n = Lp[k, n]
                b_n = float(B[k, n])

                if n not in active_nodes:
                    c_pos = np.zeros(total_vars)
                    c_pos[slot(n)] = -g_n
                    c_neg = np.zeros(total_vars)
                    c_neg[slot(n)] = g_n
                    constraints.append(_lin(c_pos, volt_max - b_n))
                    constraints.append(_lin(c_neg, volt_max + b_n))

                elif self.scheme == "forward":
                    g_nn = G_volt[k, n_next]
                    lp_nn = Lp[k, n_next]
                    b_nn = float(B[k, n_next])
                    # Left edge of segment [θ_n, θ_{n+1}]
                    gm = g_n - coupling * lp_n
                    gp = coupling * lp_n
                    for sign in (1.0, -1.0):
                        c = np.zeros(total_vars)
                        c[slot(n)] = sign * (-gm)
                        c[slot(n_next)] = sign * (-gp)
                        constraints.append(_lin(c, volt_max - sign * b_n))
                    # Right edge of the same segment
                    gmr = -coupling * lp_nn
                    gpr = g_nn + coupling * lp_nn
                    for sign in (1.0, -1.0):
                        c = np.zeros(total_vars)
                        c[slot(n)] = sign * (-gmr)
                        c[slot(n_next)] = sign * (-gpr)
                        constraints.append(_lin(c, volt_max - sign * b_nn))

                else:  # central
                    a_self = g_n
                    a_next = coupling_c * lp_n
                    a_prev = -coupling_c * lp_n
                    for sign in (1.0, -1.0):
                        c = np.zeros(total_vars)
                        c[slot(n)] = sign * (-a_self)
                        c[slot(n_next)] = sign * (-a_next)
                        c[slot(n_prev)] = sign * (-a_prev)
                        constraints.append(_lin(c, volt_max - sign * b_n))

        # Fault null-space: per-angle dynamic form, lifted to full X_flat
        idx_grid = maps["idx_grid"]  # extra_constraints_at_theta indexes fwd.vec_theta, not the coarse n_grid
        for n in range(n_grid):
            for fc in fwd.extra_constraints_at_theta(int(idx_grid[n])):
                _n = n
                _f = fc["fun"]

                def lifted(X_flat: np.ndarray, _n: int = _n, _f: Any = _f) -> float:
                    return _f(X_flat[slot(_n)])

                constraints.append({"type": "eq", "fun": lifted})

        res = minimize(
            objective,
            X_init.flatten(),
            jac=objective_grad,
            method="SLSQP",
            constraints=constraints,
            options=self.opts,
        )
        return res.x.reshape(n_grid, dim), bool(res.success)

    def minimize_current(
        self,
        torq_target: float,
        omega: float,
        guess: np.ndarray | None = None,
    ) -> Solution:
        maps = self._precompute_maps(omega)

        # R0 itself is exact regardless of this linearization -- IndependentOptimizer's
        # per-node constraints evaluate voltage at the actual candidate current, not
        # through maps["G_volt"]/["L_proj"]/["bemf_ph"].
        r0_grid, ok0 = self._run_per_angle(omega, maps, torq_target)
        if not bool(np.all(ok0)):
            return Solution(
                curr_dq=r0_grid,
                torque=float("nan"),
                success=False,
                diagnostics={
                    "r0_ok": ok0,
                    "theta_grid": maps["theta_grid"],
                    "fail_reason": "R0 failed at some nodes",
                },
            )

        # The joint solve's linear voltage constraints (_solve_joint,
        # voltage_residuals) genuinely need to stay linear-in-current for
        # tractability, unlike R0's per-node solve -- re-linearize them
        # around R0's own per-node current (a real, physically motivated
        # operating point) instead of the zeros default. Exact for
        # ConstantFlux either way; a much closer approximation than zero
        # for a current-dependent flux model (NeuralFlux).
        maps = self._precompute_maps(omega, lin_curr=r0_grid)

        X = r0_grid.copy()
        active: frozenset[int] = frozenset()
        history: list[dict[str, Any]] = []
        converged = False

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
                history.append(entry | {"action": "converged"})
                converged = True
                break

            new_active = active | violating
            if new_active == active:
                history.append(entry | {"action": "stuck"})
                break

            X_new, ok = self._solve_joint(maps, omega, torq_target, X, set(new_active))
            entry["action"] = f"solve(|A|={len(new_active)})"
            entry["solve_success"] = ok
            history.append(entry)
            X = X_new
            active = new_active

        mean_dq = np.nanmean(X, axis=0)
        torque = float(self.fwd.drive.torque(omega, mean_dq))
        return Solution(
            curr_dq=X,
            torque=torque,
            success=converged,
            diagnostics={
                "active": active,
                "history": history,
                "r0_grid": r0_grid,
                "theta_grid": maps["theta_grid"],
            },
        )

    def maximize_torque(
        self,
        omega: float,
        guess: np.ndarray | None = None,
    ) -> Solution:
        # Estimate feasible T_max from R0, then run the full joint solve at that target.
        maps = self._precompute_maps(omega)
        r0_grid, ok0 = self._run_per_angle(omega, maps, None, maximize=True)

        if not np.any(ok0):
            return Solution(
                curr_dq=np.full((self.n_grid, self.fwd.drive.dim), np.nan),
                torque=float("nan"),
                success=False,
            )

        torques = np.array([self.fwd.drive.torque(omega, r0_grid[i]) if ok0[i] else np.nan for i in range(self.n_grid)])
        t_max = float(np.nanmin(torques))
        return self.minimize_current(t_max, omega)
