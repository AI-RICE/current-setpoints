from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize

from ..models.forward_model import ForwardModel
from .dynamic_optimizer import IndependentOptimizer
from .optimizer import Solution, _run_slsqp


class FeasibilityOptimizer(IndependentOptimizer):
    """Same joint-re-solve mechanism as ``ActiveSetOptimizer``: solve every
    angle-node independently first (R0), then wherever the transition
    BETWEEN consecutive nodes isn't actually achievable given the real
    inductive voltage drop (v = R@i + omega*L@(di/dtheta) + bemf, not just
    the static v = R@i + bemf each node's own independent solve used),
    jointly re-solve those nodes with the dynamic constraint until the
    whole trajectory is consistent.

    Two changes from ``ActiveSetOptimizer`` (left untouched -- other
    experiments/tests depend on its exact existing behavior):

    1. That solve linearizes the voltage constraint ONCE, around R0,
       before its outer loop starts, and its own convergence check uses
       that same fixed linear model. That's exact for ConstantFlux
       (voltage doesn't depend on current there) but goes stale for a
       current-dependent flux model (NeuralFlux) as soon as the joint
       solve moves the current away from R0 to fix a violation --
       verified on a real case: the linear model reported converged
       (residual ~1e-10) while the TRUE voltage still peaked at 17V
       against a 13V limit on 84/90 nodes. Fixed here by re-linearizing
       around the current best trajectory every outer iteration, and
       checking convergence against the EXACT nonlinear residual.

    2. The step is bounded INSIDE the joint solve (proximal trust region)
       rather than repaired afterwards. An unbounded joint step under a
       fresh linearization overshoots far enough that the linear model is
       no longer trustworthy: measured directly on a stuck case (healthy,
       1500 RPM, 84/90 nodes active) the unbounded solve reported ZERO
       linearized violation while the EXACT residual at that same point was
       15.86 V against a 13 V limit.

       An earlier version instead damped the unbounded proposal with a
       backtracking line search, re-projecting every node back onto the
       torque target after each blend. That does not work, for two measured
       reasons. The blend direction is not a descent direction for the
       max-residual merit function -- the residual approaches the starting
       value strictly FROM ABOVE as the step shrinks, so extra backtracking
       can only waste time. And the per-node re-projection systematically
       destroys whatever the blend gained (raw 11.99 V re-projected to
       12.96 V against 12.83 V to beat), because restoring torque requires
       moving each node much further than the neighbour-consistency the
       inductive coupling will tolerate. Making that projection
       residual-aware makes it far worse (39-51 V), since omega*L/dtheta is
       a large gain and every node then cancels against a neighbour value
       that is stale the moment all nodes move at once.

       The joint solve already enforces torque as a hard equality (measured
       error ~1e-10), so each trust-region radius hands back a torque-EXACT
       trajectory directly. No blending and no re-projection are needed,
       and the coupling blow-up cannot arise. The proximal (penalty) form
       is used rather than hard box bounds because bounding the box made
       SLSQP fail outright ("positive directional derivative for
       linesearch") -- a penalty shrinks the step continuously without ever
       shrinking the feasible set.

    3. The voltage rows are ELASTIC in the restoration solve (minimize the
       worst overshoot s) instead of hard with a minimum-current objective.
       Minimizing ||X||^2 subject to linearized feasibility never tries to
       reduce the true violation at all, which is the wrong objective when
       the question being asked is whether the point is reachable. Once the
       trajectory is feasible, a final polish re-solves with the original
       hard, minimum-current formulation so the answer is still a
       minimum-current setpoint and not merely some feasible one.

    No fallback to a different optimizer anywhere in this process -- if no
    trust-region radius in the ladder improves the exact residual, this
    reports that honestly (success=False, diagnostics="stuck"/"active"), it
    does not substitute another method.
    """

    def __init__(
        self,
        fwd: ForwardModel,
        opts: dict[str, Any] | None = None,
        n_grid: int = 32,
        scheme: str = "forward",
        max_outer_iter: int = 30,
        tol: float = 1e-3,
        rho: float = 0.0,
        iron_weight: float = 0.0,
        excess_weight: float = 0.0,
        mu_ladder: tuple[float, ...] = (1.0, 0.1, 0.01, 0.001),
        strong_decrease: float = 0.3,
        smooth_window: int = 3,
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
        self.mu_ladder = mu_ladder
        self.strong_decrease = strong_decrease
        self.smooth_window = smooth_window

    def voltage_residuals_exact(
        self,
        omega: float,
        X: np.ndarray,
        idx_grid: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Max |v| - volt_max at every node, for the full dynamic voltage,
        with the static part (voltage operator/inductance/bemf) evaluated
        EXACTLY at each node's own current -- exact for any drive,
        including NeuralFlux, unlike the joint solve's linearized
        constraint (which only needs to stay linear-in-current for
        tractability inside the SLSQP call itself)."""
        n_grid, dim = X.shape
        delta_theta = 2 * np.pi / n_grid
        kept = list(self.fwd.fault._kept)
        n_surv = len(kept)

        gU_arr = np.empty((n_surv, n_grid, dim))
        gL_arr = np.empty((n_surv, n_grid, dim))
        bV_arr = np.empty((n_surv, n_grid))
        for n in range(n_grid):
            gU, gL, bV = self.fwd.volt_map_at_theta(int(idx_grid[n]), omega, X[n])
            gU_arr[:, n, :] = gU[kept]
            gL_arr[:, n, :] = gL[kept]
            bV_arr[:, n] = bV[kept]

        if self.scheme == "forward":
            dxdt = (np.roll(X, -1, axis=0) - X) / delta_theta
        else:
            dxdt = (np.roll(X, -1, axis=0) - np.roll(X, 1, axis=0)) / (2.0 * delta_theta)

        v_ph = (
            np.einsum("knj,nj->kn", gU_arr, X) + omega * np.einsum("knj,nj->kn", gL_arr, dxdt) + bV_arr
        )
        r = np.max(np.abs(v_ph), axis=0) - self.fwd.drive.volt_max
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

        for n in range(n_grid):
            for k in range(n_surv):
                h = H_curr[k, n]
                c_pos = np.zeros(total_vars)
                c_pos[slot(n)] = -h
                c_neg = np.zeros(total_vars)
                c_neg[slot(n)] = h
                constraints.append(_lin(c_pos, curr_max))
                constraints.append(_lin(c_neg, curr_max))

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
                    gm = g_n - coupling * lp_n
                    gp = coupling * lp_n
                    for sign in (1.0, -1.0):
                        c = np.zeros(total_vars)
                        c[slot(n)] = sign * (-gm)
                        c[slot(n_next)] = sign * (-gp)
                        constraints.append(_lin(c, volt_max - sign * b_n))
                    gmr = -coupling * lp_nn
                    gpr = g_nn + coupling * lp_nn
                    for sign in (1.0, -1.0):
                        c = np.zeros(total_vars)
                        c[slot(n)] = sign * (-gmr)
                        c[slot(n_next)] = sign * (-gpr)
                        constraints.append(_lin(c, volt_max - sign * b_nn))

                else:
                    a_self = g_n
                    a_next = coupling_c * lp_n
                    a_prev = -coupling_c * lp_n
                    for sign in (1.0, -1.0):
                        c = np.zeros(total_vars)
                        c[slot(n)] = sign * (-a_self)
                        c[slot(n_next)] = sign * (-a_next)
                        c[slot(n_prev)] = sign * (-a_prev)
                        constraints.append(_lin(c, volt_max - sign * b_n))

        idx_grid = maps["idx_grid"]
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

    def _project_row_to_torque(
        self,
        X_ref: np.ndarray,
        maps: dict[str, Any],
        omega: float,
        torq_target: float,
        n: int,
        is_active: bool,
    ) -> tuple[np.ndarray, bool]:
        """Closest current (Euclidean norm) to X_ref[n] that still exactly
        satisfies node n's current-limit and torque-equality constraints,
        same as _run_per_angle (R0) itself uses, just a proximity objective
        instead of minimum-norm. Used to re-project a line-search trial
        point back onto the torque target, since blending two torque-
        target-hitting currents doesn't itself hit the target (torque is
        quadratic in current).

        This proximity objective IS blind to the voltage residual, and that
        measurably costs real progress (healthy, 1500 RPM: a blended
        trajectory scoring 11.99 V re-projected to 12.96 V against 12.83 V
        to beat, turning an improving step into a rejected one). Replacing
        it with a per-node minimization of the exact dynamic residual was
        tried and is MUCH worse -- 39-51 V instead of 13-16 V. Reason: the
        dynamic voltage at node n contains omega*L*(x_{n+1} - x_n)/dtheta,
        and omega/dtheta is a large gain, so a per-node solve happily picks
        an x_n that cancels against the neighbour value it was handed. Every
        node then moves at once, every neighbour it assumed is stale, and
        the mismatch comes back multiplied by that gain. Any residual-aware
        projection has to bound how far it may move (or solve the nodes
        coupled) instead of optimizing each node freely.

        Deliberately does NOT include the per-node STATIC voltage bound
        for an ACTIVE node: that bound is exactly what the joint solve is
        allowed to relax there (in favor of the combined dynamic
        constraint, static + inductive) -- R0 already sits AT that static
        boundary for many active nodes (it's the minimum-norm point
        subject to it), so re-imposing it here would clip every trial
        step straight back to R0, silently destroying the very improvement
        the joint solve found. Non-active nodes keep the static bound,
        matching what the joint solve itself enforces for them.
        """
        fwd = self.fwd
        x_target = X_ref[n]
        H_curr_n = maps["H_curr"][:, n, :]
        fine_idx = int(maps["idx_grid"][n])
        curr_max = fwd.drive.curr_max
        volt_max = fwd.drive.volt_max

        def curr_con(x: np.ndarray) -> np.ndarray:
            return curr_max - np.abs(H_curr_n @ x)

        cons: list[dict[str, Any]] = [{"type": "ineq", "fun": curr_con}]

        if not is_active:
            H_volt_n = maps["H_volt"][:, n, :]

            def volt_con(x: np.ndarray) -> np.ndarray:
                v_dq = fwd.drive.voltage_operator(omega, x) @ x + fwd.drive.bemf_dq(omega, x)
                return volt_max - np.abs(H_volt_n @ v_dq)

            cons.append({"type": "ineq", "fun": volt_con})

        cons.append({"type": "eq", "fun": lambda x: fwd.drive.torque(omega, x) - torq_target})
        cons.extend(fwd.extra_constraints_at_theta(fine_idx))

        def obj(x: np.ndarray) -> float:
            return float(np.sum((x - x_target) ** 2))

        best_x, _, ok = _run_slsqp(obj, cons, fwd.drive.seeds(x_target), self.opts)
        return best_x, ok

    def _solve_joint_proximal(
        self,
        maps: dict[str, Any],
        omega: float,
        torq_target: float,
        X_init: np.ndarray,
        active_nodes: set[int],
        mu: float,
    ) -> tuple[np.ndarray, float, bool]:
        """Restoration step: same linearized voltage geometry as
        ``_solve_joint``, but the voltage rows are ELASTIC (relaxed by a
        single slack s, which the objective minimizes) and the step is held
        near ``X_init`` by a proximal penalty ``mu * ||X - X_init||^2``.

        Torque, current-limit and fault-achievability constraints stay HARD,
        so whatever comes back is torque-exact and current-legal by
        construction -- that is what removes the need to blend and
        re-project, and with it the neighbour-coupling failure described in
        the class docstring.

        ``mu`` is the trust-region radius in penalty form: large mu keeps
        the step short (where the linearization is still valid), mu -> 0
        recovers the unbounded solve that overshoots. Returns (X, s, ok)
        where s is the linear model's own residual claim -- which is NOT
        trustworthy on its own (at mu=0 it read 0.0 V while the exact
        residual was 15.86 V), so the caller must always score the returned
        trajectory with voltage_residuals_exact.
        """
        fwd = self.fwd
        n_grid, dim = X_init.shape
        n_surv = len(fwd.fault._kept)
        delta_theta = 2 * np.pi / n_grid
        curr_max = fwd.drive.curr_max
        volt_max = fwd.drive.volt_max
        total_vars = n_grid * dim
        coupling = omega / delta_theta
        coupling_c = omega / (2.0 * delta_theta)

        H_curr = maps["H_curr"]
        G_volt = maps["G_volt"]
        Lp = maps["L_proj"]
        B = maps["bemf_ph"]
        X0_flat = X_init.flatten()

        def slot(n: int) -> slice:
            return slice(n * dim, (n + 1) * dim)

        def objective(z: np.ndarray) -> float:
            d = z[:total_vars] - X0_flat
            return float(z[total_vars] + mu * np.dot(d, d))

        def objective_grad(z: np.ndarray) -> np.ndarray:
            g = np.zeros(total_vars + 1)
            g[:total_vars] = 2.0 * mu * (z[:total_vars] - X0_flat)
            g[total_vars] = 1.0
            return g

        constraints: list[dict[str, Any]] = []

        for n in range(n_grid):

            def eq_torq(z: np.ndarray, _n: int = n) -> float:
                return fwd.drive.torque(omega, z[slot(_n)]) - torq_target

            constraints.append({"type": "eq", "fun": eq_torq})

        def _lin_hard(coef: np.ndarray, offset: float) -> dict[str, Any]:
            def fun(z: np.ndarray) -> float:
                return float(coef @ z[:total_vars] + offset)

            def jac(z: np.ndarray) -> np.ndarray:
                g = np.zeros(total_vars + 1)
                g[:total_vars] = coef
                return g

            return {"type": "ineq", "fun": fun, "jac": jac}

        def _lin_elastic(coef: np.ndarray, offset: float) -> dict[str, Any]:
            def fun(z: np.ndarray) -> float:
                return float(coef @ z[:total_vars] + offset + z[total_vars])

            def jac(z: np.ndarray) -> np.ndarray:
                g = np.zeros(total_vars + 1)
                g[:total_vars] = coef
                g[total_vars] = 1.0
                return g

            return {"type": "ineq", "fun": fun, "jac": jac}

        for n in range(n_grid):
            for k in range(n_surv):
                h = H_curr[k, n]
                c_pos = np.zeros(total_vars)
                c_pos[slot(n)] = -h
                c_neg = np.zeros(total_vars)
                c_neg[slot(n)] = h
                constraints.append(_lin_hard(c_pos, curr_max))
                constraints.append(_lin_hard(c_neg, curr_max))

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
                    constraints.append(_lin_elastic(c_pos, volt_max - b_n))
                    constraints.append(_lin_elastic(c_neg, volt_max + b_n))

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
                        constraints.append(_lin_elastic(c, volt_max - sign * b_n))
                    gmr = -coupling * lp_nn
                    gpr = g_nn + coupling * lp_nn
                    for sign in (1.0, -1.0):
                        c = np.zeros(total_vars)
                        c[slot(n)] = sign * (-gmr)
                        c[slot(n_next)] = sign * (-gpr)
                        constraints.append(_lin_elastic(c, volt_max - sign * b_nn))

                else:
                    a_self = g_n
                    a_next = coupling_c * lp_n
                    a_prev = -coupling_c * lp_n
                    for sign in (1.0, -1.0):
                        c = np.zeros(total_vars)
                        c[slot(n)] = sign * (-a_self)
                        c[slot(n_next)] = sign * (-a_next)
                        c[slot(n_prev)] = sign * (-a_prev)
                        constraints.append(_lin_elastic(c, volt_max - sign * b_n))

        idx_grid = maps["idx_grid"]
        for n in range(n_grid):
            for fc in fwd.extra_constraints_at_theta(int(idx_grid[n])):
                constraints.append({"type": "eq", "fun": lambda z, _n=n, _f=fc["fun"]: _f(z[slot(_n)])})

        r_init, _ = self.voltage_residuals_exact(omega, X_init, idx_grid)
        s0 = max(float(np.max(r_init)) * 1.5, 1.0)
        z0 = np.concatenate([X0_flat, [s0]])
        bounds = [(-1e4, 1e4)] * total_vars + [(0.0, 1e6)]

        res = minimize(
            objective,
            z0,
            jac=objective_grad,
            method="SLSQP",
            constraints=constraints,
            bounds=bounds,
            options=self.opts,
        )
        return res.x[:total_vars].reshape(n_grid, dim), float(res.x[total_vars]), bool(res.success)

    def _smoothed_warm_start(
        self,
        X: np.ndarray,
        maps: dict[str, Any],
        omega: float,
        torq_target: float,
    ) -> tuple[np.ndarray, bool]:
        """Circular moving-average smoothing of R0's raw per-node current
        (window=self.smooth_window), then re-projecting every node back
        onto the exact torque target -- same per-node machinery R0 itself
        uses, keeping the static current/voltage bound (not yet relaxed
        for any node, since the active set isn't known until after this).

        Why: R0 solves every node completely independently, so nothing
        constrains how much current can change from one node to the next
        ("chatter") -- and the resulting inductive voltage term
        (omega*L*di/dtheta) scales directly with that chatter. On a hard
        case, chatter alone (not the machine's real physical limit) was
        the dominant driver of the exact residual (e.g. R0's per-step
        current jump of ~17A at a point whose true constant-current
        solution -- zero chatter -- sits right at the voltage boundary).
        A smoother, still torque-exact starting trajectory gives the
        joint solve/line search an easier point to correct from, without
        changing the algorithm itself or relaxing any constraint.
        """
        n_grid = X.shape[0]
        w = self.smooth_window
        if w <= 1:
            return X, True
        kernel = np.ones(w) / w
        X_smooth = np.empty_like(X)
        pad = w // 2
        for d in range(X.shape[1]):
            col = X[:, d]
            col_ext = np.concatenate([col[-pad:], col, col[:pad]]) if pad > 0 else col
            X_smooth[:, d] = np.convolve(col_ext, kernel, mode="valid")[:n_grid]

        X_proj = np.empty_like(X)
        all_ok = True
        for n in range(n_grid):
            x_proj, ok = self._project_row_to_torque(X_smooth, maps, omega, torq_target, n, is_active=False)
            X_proj[n] = x_proj
            all_ok = all_ok and ok
        return X_proj, all_ok

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
        idx_grid = maps["idx_grid"]

        r, _ = self.voltage_residuals_exact(omega, X, idx_grid)
        best_max_r = float(np.max(r))

        if best_max_r > self.tol:
            X_smooth, ok_smooth = self._smoothed_warm_start(X, maps, omega, torq_target)
            if ok_smooth:
                r_smooth, _ = self.voltage_residuals_exact(omega, X_smooth, idx_grid)
                max_r_smooth = float(np.max(r_smooth))
                if max_r_smooth < best_max_r:
                    X = X_smooth
                    r = r_smooth
                    best_max_r = max_r_smooth
                    history.append({"iter": -1, "max_V_residual": best_max_r, "action": "smoothed_warm_start"})

        for outer in range(self.max_outer_iter):
            violating = frozenset(int(n) for n in np.where(r > self.tol)[0])
            entry: dict[str, Any] = {
                "iter": outer,
                "max_V_residual": best_max_r,
                "n_violating": len(violating),
                "active_size": len(active),
            }

            if not violating:
                history.append(entry | {"action": "converged"})
                converged = True
                break

            new_active = active | violating

            maps_lin = self._precompute_maps(omega, lin_curr=X)

            best_cand: tuple[np.ndarray, np.ndarray, float, float, bool] | None = None
            for mu in self.mu_ladder:
                X_cand, s_lin, solve_ok = self._solve_joint_proximal(
                    maps_lin, omega, torq_target, X, set(new_active), mu
                )
                if not np.all(np.isfinite(X_cand)):
                    continue
                r_cand, _ = self.voltage_residuals_exact(omega, X_cand, idx_grid)
                max_r_cand = float(np.max(r_cand))
                if best_cand is None or max_r_cand < best_cand[2]:
                    best_cand = (X_cand, r_cand, max_r_cand, mu, solve_ok)
                if max_r_cand <= best_max_r - self.strong_decrease * abs(best_max_r):
                    break

            if best_cand is not None and best_cand[2] < best_max_r:
                X, r, best_max_r, mu_used, solve_ok = best_cand
                active = new_active
                entry["solve_success"] = solve_ok
                entry["mu"] = mu_used
                entry["action"] = f"accept(|A|={len(new_active)}, mu={mu_used:g})"
                history.append(entry)
            else:
                history.append(entry | {"action": "stuck"})
                break

        if converged and active:
            maps_lin = self._precompute_maps(omega, lin_curr=X)
            X_polish, polish_ok = self._solve_joint(maps_lin, omega, torq_target, X, set(active))
            if polish_ok and np.all(np.isfinite(X_polish)):
                r_polish, _ = self.voltage_residuals_exact(omega, X_polish, idx_grid)
                max_r_polish = float(np.max(r_polish))
                norm_before = float(np.sum(X**2))
                norm_after = float(np.sum(X_polish**2))
                if max_r_polish <= self.tol and norm_after < norm_before:
                    X = X_polish
                    best_max_r = max_r_polish
                    history.append(
                        {
                            "action": "min_current_polish",
                            "max_V_residual": max_r_polish,
                            "curr_norm_before": norm_before,
                            "curr_norm_after": norm_after,
                        }
                    )

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
