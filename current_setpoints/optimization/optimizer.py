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

        def curr_con(i: np.ndarray) -> float:
            return fwd.drive.curr_max - fwd.peak_vals(omega, i)[0]

        def volt_con(i: np.ndarray) -> float:
            return fwd.drive.volt_max - fwd.peak_vals(omega, i)[1]

        cons: list[dict[str, Any]] = [
            {"type": "ineq", "fun": curr_con},
            {"type": "ineq", "fun": volt_con},
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


class HullConstrainedOptimizer(StaticOptimizer):
    """StaticOptimizer that additionally confines every setpoint (and the
    T_max probe) to the drive's sampled-current convex hull via
    `drive.hull_margins`. For LUT / MLP flux models this guarantees the flux
    map is only ever interpolated, never extrapolated: cells whose optimum
    would leave the hull come back infeasible (honest holes) instead of
    returning extrapolated — and, as seen for the Tesla5f MLP, physically
    wrong — torque. No-op for drives without `hull_margins`."""

    def _base_constraints(self, omega: float) -> list[dict[str, Any]]:
        cons = super()._base_constraints(omega)
        drive = self.fwd.drive
        if hasattr(drive, "hull_margins"):
            cons.append({"type": "ineq", "fun": drive.hull_margins})
        return cons


def _positive_harmonics(harmonics: tuple[int, ...]) -> list[int]:
    return [int(h) for h in sorted(set(harmonics)) if int(h) > 0]


def _n_basis(harmonics: tuple[int, ...]) -> int:
    has_dc = 0 in {int(h) for h in harmonics}
    return (1 if has_dc else 0) + 2 * len(_positive_harmonics(harmonics))


def _fourier_design(theta: np.ndarray, harmonics: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    theta = np.asarray(theta, dtype=float)
    cols: list[np.ndarray] = []
    dcols: list[np.ndarray] = []
    if 0 in {int(h) for h in harmonics}:
        cols.append(np.ones_like(theta))
        dcols.append(np.zeros_like(theta))
    for h in _positive_harmonics(harmonics):
        cols.append(np.cos(h * theta))
        dcols.append(-h * np.sin(h * theta))
        cols.append(np.sin(h * theta))
        dcols.append(h * np.cos(h * theta))
    return np.column_stack(cols), np.column_stack(dcols)


def _parseval_weights(harmonics: tuple[int, ...]) -> np.ndarray:
    w: list[float] = []
    if 0 in {int(h) for h in harmonics}:
        w.append(1.0)
    for _ in _positive_harmonics(harmonics):
        w.extend([0.5, 0.5])
    return np.asarray(w, dtype=float)


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
        for n in range(n_grid):
            for fc in fwd.extra_constraints_at_theta(n):
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


class FourierOptimizer(DynamicOptimizer):
    def __init__(
        self,
        fwd: ForwardModel,
        opts: dict[str, Any] | None = None,
        n_grid: int = 180,
        n_out: int = 256,
        harmonics: tuple[int, ...] = (0, 9, 11),
        ripple_budget: float = 1e-2,
    ) -> None:
        super().__init__(
            fwd,
            opts if opts is not None else {"disp": False, "ftol": 1e-9, "maxiter": 2000, "eps": 1e-8},
            n_grid,
        )
        if 0 not in {int(h) for h in harmonics}:
            raise ValueError("harmonics must include the DC term 0.")
        self.harmonics: tuple[int, ...] = tuple(sorted(int(h) for h in harmonics))
        self.n_out = n_out
        self.ripple_budget = ripple_budget

    def _build_constraints(
        self,
        maps: dict[str, Any],
        omega: float,
        torq_target: float | None,
        A_t: np.ndarray,
        b_t: np.ndarray,
        c_t: float,
        Phi: np.ndarray,
        dPhi: np.ndarray,
        nb: int,
        dim: int,
    ) -> list[dict[str, Any]]:
        fwd = self.fwd
        H_curr = maps["H_curr"]  # (n_surv, n_grid, dim)
        G_volt = maps["G_volt"]  # (n_surv, n_grid, dim)
        Lp = maps["L_proj"]  # (n_surv, n_grid, dim)
        B = maps["bemf_ph"]  # (n_surv, n_grid)
        n_con = self.n_grid
        n_surv = H_curr.shape[0]
        curr_max = fwd.drive.curr_max
        volt_max = fwd.drive.volt_max

        def as_C(c: np.ndarray) -> np.ndarray:
            return c.reshape(nb, dim)

        def torque_at(c: np.ndarray) -> np.ndarray:
            i_s = Phi @ as_C(c)
            return np.einsum("sj,jk,sk->s", i_s, A_t, i_s) + i_s @ b_t + c_t

        def torque_grad_rows(c: np.ndarray) -> np.ndarray:
            i_s = Phi @ as_C(c)
            g_i = 2.0 * (i_s @ A_t) + b_t[None, :]
            return (Phi[:, :, None] * g_i[:, None, :]).reshape(n_con, nb * dim)

        def _lin(coef: np.ndarray, offset: float) -> dict[str, Any]:
            def fun(c: np.ndarray) -> float:
                return float(coef @ c + offset)

            def jac(c: np.ndarray) -> np.ndarray:
                return coef

            return {"type": "ineq", "fun": fun, "jac": jac}

        constraints: list[dict[str, Any]] = []

        # Torque: mean equality + pointwise ripple band (minimize_current only)
        if torq_target is not None:

            def eq_mean(c: np.ndarray) -> float:
                return float(np.mean(torque_at(c)) - torq_target)

            def eq_mean_jac(c: np.ndarray) -> np.ndarray:
                return torque_grad_rows(c).mean(axis=0)

            constraints.append({"type": "eq", "fun": eq_mean, "jac": eq_mean_jac})

            for s in range(n_con):
                phi_s = Phi[s]
                for sign in (1.0, -1.0):

                    def rip_fun(c: np.ndarray, _s: int = s, _sg: float = sign, _phi: np.ndarray = phi_s) -> float:
                        i_sv = _phi @ as_C(c)
                        T_s = float(i_sv @ A_t @ i_sv + i_sv @ b_t + c_t)
                        return self.ripple_budget - _sg * (T_s - torq_target)

                    def rip_jac(c: np.ndarray, _s: int = s, _sg: float = sign, _phi: np.ndarray = phi_s) -> np.ndarray:
                        i_sv = _phi @ as_C(c)
                        return -_sg * np.outer(_phi, 2.0 * (A_t @ i_sv) + b_t).ravel()

                    constraints.append({"type": "ineq", "fun": rip_fun, "jac": rip_jac})

        # Current: two-sided linear
        for s in range(n_con):
            for k in range(n_surv):
                coef_i = np.outer(Phi[s], H_curr[k, s]).ravel()
                constraints.append(_lin(-coef_i, curr_max))
                constraints.append(_lin(coef_i, curr_max))

        # Voltage: two-sided linear, exact inductive term via Fourier derivative
        # v_k(θ_s) = G_volt[k,s] @ i_s + ω · L_proj[k,s] @ (di_s/dθ) + bemf[k,s]
        for s in range(n_con):
            for k in range(n_surv):
                coef_v = (np.outer(Phi[s], G_volt[k, s]) + omega * np.outer(dPhi[s], Lp[k, s])).ravel()
                b_kn = float(B[k, s])
                constraints.append(_lin(-coef_v, volt_max - b_kn))
                constraints.append(_lin(coef_v, volt_max + b_kn))

        # Fault null-space: N·R(θ_s)·i(θ_s) = 0, lifted to Fourier coefficients
        for s in range(n_con):
            for fc in fwd.extra_constraints_at_theta(s):
                _phi_s = Phi[s]
                _f = fc["fun"]

                def fault_eq(c: np.ndarray, _phi: np.ndarray = _phi_s, _f: Any = _f) -> float:
                    return _f(_phi @ as_C(c))

                constraints.append({"type": "eq", "fun": fault_eq})

        return constraints

    def _solve(
        self,
        omega: float,
        maps: dict[str, Any],
        torq_target: float | None,
        warm_coeffs: np.ndarray | None = None,
    ) -> dict[str, Any]:
        fwd = self.fwd
        dim = fwd.drive.dim
        harmonics = self.harmonics
        nb = _n_basis(harmonics)
        n_con = self.n_grid
        maximize = torq_target is None

        theta_s = fwd.vec_theta[maps["idx_grid"]]
        Phi, dPhi = _fourier_design(theta_s, harmonics)
        A_t, b_t, c_t = fwd.drive.torque_quadratic(omega)
        wvec = np.repeat(_parseval_weights(harmonics), dim)

        def as_C(c: np.ndarray) -> np.ndarray:
            return c.reshape(nb, dim)

        if maximize:

            def objective(c: np.ndarray) -> float:
                i_s = Phi @ as_C(c)
                return -float(np.mean(np.einsum("sj,jk,sk->s", i_s, A_t, i_s) + i_s @ b_t + c_t))

            def objective_grad(c: np.ndarray) -> np.ndarray:
                i_s = Phi @ as_C(c)
                g_i = -(2.0 * (i_s @ A_t) + b_t[None, :]) / n_con
                return (Phi[:, :, None] * g_i[:, None, :]).reshape(n_con, nb * dim).sum(axis=0)
        else:

            def objective(c: np.ndarray) -> float:
                return float(np.sum(wvec * c * c))

            def objective_grad(c: np.ndarray) -> np.ndarray:
                return 2.0 * wvec * c

        constraints = self._build_constraints(maps, omega, torq_target, A_t, b_t, c_t, Phi, dPhi, nb, dim)

        # Warm start: DC-only pre-solve when AC harmonics are present.
        # Uses StaticOptimizer rather than a recursive _solve call (recursive call
        # with the same harmonics caused infinite recursion when nb > dim).
        C0 = np.zeros((nb, dim))
        if warm_coeffs is not None:
            C0[: min(nb, warm_coeffs.shape[0])] = warm_coeffs[:nb]
        elif nb > dim and not maximize:
            from .optimizer import StaticOptimizer as _SO

            dc_sol = _SO(fwd, self.opts).minimize_current(torq_target, omega)
            C0[0] = dc_sol.curr_dq if dc_sol.success else fwd.drive.seeds()[0]
        else:
            C0[0] = fwd.drive.seeds()[0]

        res = minimize(
            objective,
            C0.ravel(),
            jac=objective_grad,
            method="SLSQP",
            constraints=constraints,
            options=self.opts,
        )
        C = res.x.reshape(nb, dim)

        theta_out = np.linspace(0.0, 2 * np.pi, self.n_out, endpoint=False)
        Phi_out, _ = _fourier_design(theta_out, harmonics)
        curr_dq_grid = Phi_out @ C

        torques_con = np.einsum("sj,jk,sk->s", Phi @ C, A_t, Phi @ C) + (Phi @ C) @ b_t + c_t
        torque_mean = float(np.mean(torques_con))
        torque_ripple = float(np.max(np.abs(torques_con - torque_mean)))

        return {
            "C": C,
            "curr_dq_grid": curr_dq_grid,
            "theta_out": theta_out,
            "success": bool(res.success),
            "joule": float(np.sum(wvec * res.x**2)),
            "torque_mean": torque_mean,
            "torque_ripple": torque_ripple,
            "n_basis": nb,
            "harmonics": harmonics,
        }

    def minimize_current(
        self,
        torq_target: float,
        omega: float,
        guess: np.ndarray | None = None,
    ) -> Solution:
        maps = self._precompute_maps(omega)
        warm = None
        if guess is not None and guess.ndim == 1:
            nb = _n_basis(self.harmonics)
            warm = np.zeros((nb, self.fwd.drive.dim))
            warm[0] = guess
        result = self._solve(omega, maps, torq_target, warm_coeffs=warm)
        return Solution(
            curr_dq=result["curr_dq_grid"],
            torque=torq_target,
            success=result["success"],
            diagnostics={k: v for k, v in result.items() if k != "curr_dq_grid"},
        )

    def maximize_torque(
        self,
        omega: float,
        guess: np.ndarray | None = None,
    ) -> Solution:
        maps = self._precompute_maps(omega)
        result = self._solve(omega, maps, torq_target=None)
        return Solution(
            curr_dq=result["curr_dq_grid"],
            torque=result["torque_mean"],
            success=result["success"],
            diagnostics={k: v for k, v in result.items() if k != "curr_dq_grid"},
        )
