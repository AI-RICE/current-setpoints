from __future__ import annotations

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
    ) -> None:
        super().__init__(
            fwd,
            opts if opts is not None else {"disp": False, "ftol": 1e-7, "maxiter": 1500, "eps": 1e-8},
            n_grid,
        )
        if scheme not in ("forward", "central"):
            raise ValueError(f"scheme must be 'forward' or 'central', got {scheme!r}")
        self.scheme = scheme
        # Default N (= n_grid): the PDF's bound on outer iterations of Algorithm 1.
        self.max_outer_iter = max_outer_iter if max_outer_iter is not None else n_grid
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
        v_abs = np.max(np.abs(v_ph), axis=0)
        if self.scheme == "forward":
            # v_ph is the voltage at the START of segment n (PDF (9d)). The current runs linearly
            # from i_n to i_{n+1}, so the voltage at the END of segment n (theta_{n+1}, same slope)
            # must be within the limit too -- the second voltage _solve_joint limits for active
            # nodes. r_n covers both ends of segment n, so activating node n constrains both.
            v_end = np.roll(np.einsum("knj,nj->kn", G, X) + B, -1, axis=1) + omega * np.einsum(
                "knj,nj->kn", np.roll(Lp, -1, axis=1), dxdt
            )
            v_abs = np.maximum(v_abs, np.max(np.abs(v_end), axis=0))
        r = v_abs - self.fwd.drive.volt_max
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

        # Per-unit scaling of what SLSQP sees -- problem (9) itself is unchanged: dividing the
        # objective, an inequality or an equality by a positive constant changes neither the
        # feasible set nor the minimizer. SLSQP is not scale-invariant: unscaled, the objective
        # (~5e4 at 60 nodes) dwarfs the constraint rows (~1-10 V or A) and its line search failed
        # ("Positive directional derivative") at random points near max torque at high speed.
        obj_scale = 1.0 / (n_grid * curr_max**2)
        torq_scale = max(abs(torq_target), 1.0)

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

        # Torque equalities (nonlinear). Exact gradient when the drive provides one (PMSMDrive.
        # torque_grad) -- same constraint, only its slope is exact instead of SLSQP's finite
        # differences, which made SLSQP's line search fail ("Positive directional derivative")
        # with 60 nodes near max torque at high speed. Other drives: finite differences as before.
        torque_grad = getattr(fwd.drive, "torque_grad", None)
        for n in range(n_grid):

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

        def _lin(coef: np.ndarray, offset: float, scale: float) -> dict[str, Any]:
            # coef @ x + offset >= 0, divided by its limit (curr_max or volt_max): per-unit row
            coef_s, offset_s = coef / scale, offset / scale

            def fun(X_flat: np.ndarray) -> float:
                return float(coef_s @ X_flat + offset_s)

            def jac(X_flat: np.ndarray) -> np.ndarray:
                return coef_s

            return {"type": "ineq", "fun": fun, "jac": jac}

        # Current: always static (no inductive coupling)
        for n in range(n_grid):
            for k in range(n_surv):
                h = H_curr[k, n]
                c_pos = np.zeros(total_vars)
                c_pos[slot(n)] = -h
                c_neg = np.zeros(total_vars)
                c_neg[slot(n)] = h
                constraints.append(_lin(c_pos, curr_max, curr_max))
                constraints.append(_lin(c_neg, curr_max, curr_max))

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
                    constraints.append(_lin(c_pos, volt_max - b_n, volt_max))
                    constraints.append(_lin(c_neg, volt_max + b_n, volt_max))

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
                        constraints.append(_lin(c, volt_max - sign * b_n, volt_max))
                    # Right edge of the same segment
                    gmr = -coupling * lp_nn
                    gpr = g_nn + coupling * lp_nn
                    for sign in (1.0, -1.0):
                        c = np.zeros(total_vars)
                        c[slot(n)] = sign * (-gmr)
                        c[slot(n_next)] = sign * (-gpr)
                        constraints.append(_lin(c, volt_max - sign * b_nn, volt_max))

                else:  # central
                    a_self = g_n
                    a_next = coupling_c * lp_n
                    a_prev = -coupling_c * lp_n
                    for sign in (1.0, -1.0):
                        c = np.zeros(total_vars)
                        c[slot(n)] = sign * (-a_self)
                        c[slot(n_next)] = sign * (-a_next)
                        c[slot(n_prev)] = sign * (-a_prev)
                        constraints.append(_lin(c, volt_max - sign * b_n, volt_max))

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
        # Nodes where X breaks a non-voltage constraint: torque equality, current limit,
        # or fault null-space. Voltage is checked separately (voltage_residuals), since
        # in the active set it is the coupled dynamic form, not a per-node one.
        fwd = self.fwd
        curr_max = fwd.drive.curr_max
        bad = []
        for n in range(X.shape[0]):
            H_curr_n = maps["H_curr"][:, n, :]
            cons: list[dict[str, Any]] = [
                {"type": "eq", "fun": lambda x: fwd.drive.torque(omega, x) - torq_target},
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

        # R0 itself is exact regardless of this linearization -- IndependentOptimizer's
        # per-node constraints evaluate voltage at the actual candidate current, not
        # through maps["G_volt"]/["L_proj"]/["bemf_ph"].
        r0_grid, ok0 = self._run_per_angle(omega, maps, torq_target, guess=guess)
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
        fail_reason: str | None = None

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
                # Voltage alone is not enough: converged only if X also meets torque,
                # current and fault constraints at every node.
                bad_nodes = self._nodes_violating_static(maps, omega, torq_target, X)
                if bad_nodes:
                    history.append(entry | {"action": "voltage ok, other constraints violated", "bad_nodes": bad_nodes})
                    fail_reason = f"torque/current/fault constraint violated at {len(bad_nodes)} nodes"
                else:
                    history.append(entry | {"action": "converged"})
                    converged = True
                break

            # PDF Algorithm 1 stops only when A stops growing AND all r_n <= 0 -- if A
            # did not grow but nodes still violate, re-solve. For a linear machine this
            # cannot happen; for NeuralFlux it re-solves around the updated linearization.
            new_active = active | violating

            X_new, ok = self._solve_joint(maps, omega, torq_target, X, set(new_active))
            entry["action"] = f"solve(|A|={len(new_active)})"
            entry["solve_success"] = ok
            history.append(entry)
            if not ok:
                # Reject a failed joint solve: keep the previous X rather than passing
                # a possibly constraint-violating point on as the next linearization.
                fail_reason = f"joint solve failed at outer iteration {outer}"
                break
            X = X_new
            active = new_active
            # Re-linearize around the NEW X before the next residual check/solve --
            # otherwise every later iteration keeps checking (and solving) against
            # the voltage model evaluated at the stale r0_grid point, and "converged"
            # would only mean "satisfies that stale linearization," not the actual
            # current-dependent (NeuralFlux) voltage at the real solution. Since
            # G_volt/L_proj/bemf_ph are exactly U(lin_curr)/L(lin_curr)/bemf(lin_curr)
            # -- the same formula waveforms_for_plot uses for the "exact" voltage --
            # linearizing around X makes the next check exact AT that X, not approximate.
            maps = self._precompute_maps(omega, lin_curr=X)
        else:
            fail_reason = f"max_outer_iter={self.max_outer_iter} reached"

        # Mean of the torques actually delivered at each node, converged or not --
        # never assumed equal to torq_target (the last joint solve may have failed).
        node_torques = self._node_torques(omega, X)
        return Solution(
            curr_dq=X,
            torque=float(np.nanmean(node_torques)),
            success=converged,
            diagnostics={
                "active": active,
                "history": history,
                "r0_grid": r0_grid,
                "theta_grid": maps["theta_grid"],
                "torque_min": float(np.nanmin(node_torques)),
                "torque_max": float(np.nanmax(node_torques)),
                "fail_reason": fail_reason,
            },
        )

    # 9 steps: final bracket t_naive / 2**9 (~0.015 Nm here); 5 steps left ~0.2 Nm, enough
    # for the reported dynamic T_max to land below the static one it provably reaches.
    _N_BISECT_ITERS = 9

    def maximize_torque(
        self,
        omega: float,
        guess: np.ndarray | None = None,
        start_margin: float = 1.0,
    ) -> Solution:
        # R0's raw per-node estimate (min of 60 independently-maximized, mutually
        # unconstrained torques) systematically overestimates what one current
        # trajectory can hit at all 60 angles simultaneously -- and by how MUCH
        # varies with omega: R0's per-node problems are solved independently, with
        # no inductive coupling term at all (that term is ~omega/delta_theta), so
        # the overestimate is small at low speed (coupling negligible) and large at
        # high speed (coupling dominant). A FIXED backoff fraction that works at one
        # speed can therefore fail at another purely because the reference itself
        # scales differently, not because of any solver/iteration-budget issue.
        # Bisect between 0 (never achievable... unless it is) and the raw estimate
        # instead of guessing fixed fractions -- same bounded number of solve
        # attempts (1 + _N_BISECT_ITERS), but finds wherever the true ceiling
        # actually is at THIS speed, whatever fraction of t_naive that turns out
        # to be.
        #
        # start_margin: for a sweep over many (omega, target) points, neighboring
        # points tend to need a similar fraction -- pass the previous point's
        # sol.diagnostics["margin"] here as the first guess, then bisection still
        # refines from there instead of accepting it blindly.
        maps = self._precompute_maps(omega)
        r0_grid, ok0 = self._run_per_angle(omega, maps, None, maximize=True, guess=guess)

        if not np.any(ok0):
            return Solution(
                curr_dq=np.full((self.n_grid, self.fwd.drive.dim), np.nan),
                torque=float("nan"),
                success=False,
            )

        torques = np.array([self.fwd.drive.torque(omega, r0_grid[i]) if ok0[i] else np.nan for i in range(self.n_grid)])
        t_naive = float(np.nanmin(torques))

        lo, hi = 0.0, t_naive
        best_sol: Solution | None = None
        best_margin = 0.0

        # Fast-path guess using the carried-forward margin -- if it succeeds, it
        # gives an early lo bound (fewer bisection steps needed below). If it
        # FAILS, that says nothing about whether something between it and t_naive
        # is achievable at THIS speed (a neighboring, easier speed's margin is not
        # a valid upper bound here) -- discard it and let real bisection search
        # the full, always-correct [0, t_naive] range instead of narrowing hi to
        # the failed guess (that bug caused a monotonic hi-collapse across a sweep:
        # one low margin, once carried forward, permanently traps every later
        # probe's search below it).
        sol = self.minimize_current(start_margin * t_naive, omega, guess=guess)
        if sol.success:
            lo = start_margin * t_naive
            best_sol, best_margin = sol, start_margin

        for _ in range(self._N_BISECT_ITERS):
            if lo >= hi:  # first try already reached t_naive -- nothing left to bisect
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
