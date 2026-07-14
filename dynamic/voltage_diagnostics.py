"""
Successive-relaxation / active-set solver for the dynamic problem.

Implements the algorithm in ``Dynamic.tex`` Section 4. Starting from the
R0 trajectory (independent per-angle, no coupling), we:

1. Evaluate the *full* dynamic voltage including the inductive term
   ``omega * L_s * (x_{n+1} - x_n) / Delta_theta`` at every node and at
   every phase.
2. Collect the nodes where the worst-phase voltage exceeds ``V_max``
   into the active set ``A``.
3. Re-solve the *joint* trajectory problem with the dynamic voltage
   constraint enforced on ``A`` and the cheap static voltage constraint
   on the other nodes.
4. Iterate, growing ``A`` monotonically until no node violates the
   dynamic constraint.

The joint solve is one SLSQP problem in ``N * dim`` variables with
``N`` torque equalities and ``N * n_phases * 2`` two-sided current and
voltage inequalities. We supply analytical Jacobians for the linear
inequalities to keep SLSQP fast.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize

from current_setpoints.optimization.models import BaseTorqueModel
from current_setpoints.simulation import Transform
from dynamic.independent_optimizer import run_independent_per_angle


def _grid_indices(transform: Transform, n_grid: int) -> tuple[np.ndarray, np.ndarray]:
    """Map the coarse angle grid to integer indices into ``transform.vec_theta``."""
    n_theta = transform.vec_theta.size - 1
    theta_grid = np.linspace(0.0, 2 * np.pi, n_grid, endpoint=False)
    idx_grid = (np.round(theta_grid / (2 * np.pi) * n_theta).astype(int)) % n_theta
    return theta_grid, idx_grid


def _phase_basis(transform: Transform, idx_grid: np.ndarray) -> np.ndarray:
    """``h_k(theta_n)`` rows for each phase ``k`` and node ``n``; shape ``(n_phases, n_grid, dim)``."""
    n_theta = transform.vec_theta.size - 1
    shift = transform._phase_shift_samples
    return np.stack(
        [transform.mat_dq_to_ph[(idx_grid - k * shift) % n_theta] for k in range(transform.n_phases)],
        axis=0,
    )


def voltage_residuals_dense(
    transform: Transform,
    omega: float,
    curr_dq_grid: np.ndarray,
    volt_max: float,
) -> tuple[float, np.ndarray]:
    """
    Worst-case dynamic voltage residual on the *dense* angle grid used by
    ``plot_dq_phase_combined`` (700 samples). This is what the figure
    shows. It can exceed the coarse-grid SLSQP residual because the static
    voltage contribution drifts linearly within each segment while the
    inductive term is constant (forward Euler).

    Returns ``(max_residual, v_ph_dense)``: scalar worst-phase residual
    over all dense angles, and the dense per-phase voltage array of shape
    ``(n_phases, n_theta + 1)``.
    """
    from current_setpoints.utils.plotting_dynamic import (
        _dynamic_phase_waveforms,
        _phase_currents_all,
    )

    n_grid = curr_dq_grid.shape[0]
    theta_grid = np.linspace(0.0, 2 * np.pi, n_grid, endpoint=False)
    _, v_ph_a = _dynamic_phase_waveforms(transform, omega, theta_grid, curr_dq_grid)
    v_ph_dense = _phase_currents_all(transform, v_ph_a)
    return float(np.max(np.abs(v_ph_dense)) - volt_max), v_ph_dense


def voltage_residuals(
    transform: Transform,
    omega: float,
    curr_dq_grid: np.ndarray,
    volt_max: float,
    scheme: str = "forward",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Worst-phase dynamic voltage residual at every coarse node (where SLSQP
    enforces it). Uses the same finite-difference ``scheme`` as the solver
    so the active-set update is consistent.

    Returns ``(r, v_ph_grid)`` where ``r[n] = max_k |v_k(theta_n)| - V_max``
    and ``v_ph_grid`` has shape ``(n_grid, n_phases)``.
    """
    transform._set_omega(omega)
    n_grid, dim = curr_dq_grid.shape
    delta_theta = 2 * np.pi / n_grid

    if scheme == "forward":
        dxdtheta = (np.roll(curr_dq_grid, -1, axis=0) - curr_dq_grid) / delta_theta
    elif scheme == "central":
        dxdtheta = (np.roll(curr_dq_grid, -1, axis=0) - np.roll(curr_dq_grid, 1, axis=0)) / (2.0 * delta_theta)
    else:
        raise ValueError(f"scheme must be 'forward' or 'central', got {scheme!r}")

    U = transform.mat_curr_dq_to_volt_dq
    L = transform.machine.L_stat
    flux_volt, _ = transform.flux.get_flux(omega, np.zeros(dim))
    bemf_dq = omega * transform.machine.mat_crossc @ flux_volt

    v_dq_grid = curr_dq_grid @ U.T + omega * dxdtheta @ L.T + bemf_dq[None, :]

    _, idx_grid = _grid_indices(transform, n_grid)
    H = _phase_basis(transform, idx_grid)
    v_ph_grid = np.einsum("knj,nj->nk", H, v_dq_grid)

    r = np.max(np.abs(v_ph_grid), axis=1) - volt_max
    return r, v_ph_grid


def _solve_joint(
    model: BaseTorqueModel,
    transform: Transform,
    omega: float,
    torq_target: float,
    curr_max: float,
    volt_max: float,
    X_init: np.ndarray,
    active_nodes: set[int],
    opts: dict[str, Any],
    rho: float = 0.0,
    scheme: str = "forward",
    iron_weight: float = 0.0,
    excess_weight: float = 0.0,
) -> tuple[np.ndarray, bool]:
    """
    Joint SLSQP solve for the trajectory ``X = (x_0, ..., x_{N-1})`` with
    pointwise torque, current and voltage limits. Voltage limit is the
    *dynamic* (coupled) form for nodes in ``active_nodes`` and the static
    form otherwise.
    """
    transform._set_omega(omega)
    n_grid, dim = X_init.shape
    n_phases = transform.n_phases
    delta_theta = 2 * np.pi / n_grid

    _, idx_grid = _grid_indices(transform, n_grid)
    H = _phase_basis(transform, idx_grid)  # (n_phases, n_grid, dim)

    U = transform.mat_curr_dq_to_volt_dq
    L = transform.machine.L_stat
    flux_volt, _ = transform.flux.get_flux(omega, np.zeros(dim))
    bemf_dq = omega * transform.machine.mat_crossc @ flux_volt

    G = np.einsum("knj,jl->knl", H, U)  # (n_phases, n_grid, dim) -- static voltage proj
    bemf_ph = np.einsum("knj,j->kn", H, bemf_dq)  # (n_phases, n_grid)       -- BEMF projection
    L_proj = np.einsum("knj,jl->knl", H, L)  # (n_phases, n_grid, dim) -- inductive proj

    coupling = omega / delta_theta

    def slot(n: int) -> slice:
        return slice(n * dim, (n + 1) * dim)

    # Pre-compute iron-loss objective terms when either weight is positive.
    # Eddy:    P_e = (omega^2 / dtheta^2 / N) * sum_n sum_k (Δλ_k^n)^2
    # Bertotti: P_x = (1 / N) * sum_n sum_k |omega * Δλ_k^n / dtheta|^1.5
    # Both share the linear projection A_{k,n} = h_k(θ_n) L and bias
    # bias_{k,n} = (h_k(θ_{n+1}) - h_k(θ_n)) Ψ_PM.
    iron_active = iron_weight > 0.0 or excess_weight > 0.0
    if iron_active:
        L_stat = transform.machine.L_stat
        A_per_node = np.einsum("knj,jl->knl", H, L_stat)
        psi_pm_proj = np.einsum("knj,j->kn", H, flux_volt)
        iron_bias = np.roll(psi_pm_proj, -1, axis=1) - psi_pm_proj
        iron_scale = (omega**2) / (delta_theta**2) / n_grid
        # Bertotti scale: |omega * delta/dtheta|^1.5 = (omega/dtheta)^1.5 * |delta|^1.5
        excess_scale = (omega / delta_theta) ** 1.5 / n_grid

    def _iron_term_and_grad(X_flat: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Combined iron-loss objective:
        ``iron_weight * P_eddy + excess_weight * P_bertotti`` and its
        gradient. Returns (0, 0) if both weights are zero.
        """
        if not iron_active:
            return 0.0, np.zeros_like(X_flat)
        X = X_flat.reshape(n_grid, dim)
        proj = np.einsum("knj,nj->kn", A_per_node, X)
        delta = np.roll(proj, -1, axis=1) - proj + iron_bias

        val = 0.0
        # weight_per_delta[k, n] = d (cost) / d (delta[k, n])
        weight_per_delta = np.zeros_like(delta)
        if iron_weight > 0.0:
            val += iron_weight * iron_scale * float(np.sum(delta**2))
            weight_per_delta += 2.0 * iron_weight * iron_scale * delta
        if excess_weight > 0.0:
            abs_delta = np.abs(delta)
            val += excess_weight * excess_scale * float(np.sum(abs_delta**1.5))
            # d/dx |x|^1.5 = 1.5 sign(x) |x|^0.5
            weight_per_delta += 1.5 * excess_weight * excess_scale * np.sign(delta) * np.sqrt(abs_delta)
        # ∂delta[k,m]/∂X[n,j] = A_{k,n,j} * ((n == (m+1)%N) - (n == m))
        weight_diff = np.roll(weight_per_delta, 1, axis=1) - weight_per_delta
        grad_X = np.einsum("knj,kn->nj", A_per_node, weight_diff)
        return val, grad_X.flatten()

    if rho > 0.0:
        slew_idx_pairs = [(n, (n + 1) % n_grid) for n in range(n_grid)]

        def objective(X_flat: np.ndarray) -> float:
            base = float(np.sum(X_flat**2))
            slew = 0.0
            for n_, np_ in slew_idx_pairs:
                d = X_flat[slot(n_)] - X_flat[slot(np_)]
                slew += float(np.dot(d, d))
            iron_v, _ = _iron_term_and_grad(X_flat)
            return base + rho * slew + iron_v

        def objective_grad(X_flat: np.ndarray) -> np.ndarray:
            g = 2.0 * X_flat.copy()
            for n_, np_ in slew_idx_pairs:
                d = X_flat[slot(n_)] - X_flat[slot(np_)]
                g[slot(n_)] += 2.0 * rho * d
                g[slot(np_)] -= 2.0 * rho * d
            _, ig = _iron_term_and_grad(X_flat)
            g += ig
            return g
    else:

        def objective(X_flat: np.ndarray) -> float:
            iron_v, _ = _iron_term_and_grad(X_flat)
            return float(np.sum(X_flat**2)) + iron_v

        def objective_grad(X_flat: np.ndarray) -> np.ndarray:
            _, ig = _iron_term_and_grad(X_flat)
            return 2.0 * X_flat + ig

    constraints: list[dict[str, Any]] = []

    # Torque equalities, one per node (nonlinear; rely on SLSQP finite-diff gradient).
    for n in range(n_grid):

        def eq_torq(X_flat: np.ndarray, _n: int = n) -> float:
            return model.calculate_torque(omega, X_flat[slot(_n)]) - torq_target

        constraints.append({"type": "eq", "fun": eq_torq})

    total_vars = n_grid * dim

    def _make_lin(
        coef_self: np.ndarray,
        n: int,
        sign: float,
        offset: float,
        coef_next: np.ndarray | None = None,
        n_next: int | None = None,
    ):
        """Build a linear inequality ``sign * (coef_self . x_n + coef_next . x_{n+1} + offset) >= 0``."""
        full = np.zeros(total_vars)
        full[slot(n)] = sign * coef_self
        if coef_next is not None and n_next is not None:
            full[slot(n_next)] += sign * coef_next
        c0 = sign * offset

        def fun(X_flat: np.ndarray) -> float:
            return float(full @ X_flat + c0)

        def jac(X_flat: np.ndarray) -> np.ndarray:
            return full

        return {"type": "ineq", "fun": fun, "jac": jac}

    # Current inequalities (uncoupled, always static form -- they don't depend on omega*dx/dtheta).
    for n in range(n_grid):
        for k in range(n_phases):
            h_kn = H[k, n]
            # curr_max - h @ x >= 0
            constraints.append(_make_lin(coef_self=-h_kn, n=n, sign=1.0, offset=curr_max))
            # curr_max + h @ x >= 0
            constraints.append(_make_lin(coef_self=h_kn, n=n, sign=1.0, offset=curr_max))

    # Voltage inequalities. Two discretisation choices for the inductive
    # coupling, switched by ``scheme``:
    #   - "forward": forward-Euler difference (x_{n+1} - x_n)/dtheta.
    #     The inductive term is constant within each segment and
    #     discontinuous at every node. Constraints are enforced at BOTH
    #     edges of segment [theta_n, theta_{n+1}] to suppress
    #     within-segment drift in the piecewise-linear reconstruction.
    #   - "central": centred difference (x_{n+1} - x_{n-1})/(2 dtheta).
    #     The inductive term varies smoothly across nodes, removing the
    #     jumps; we then enforce only at the node, matching the
    #     reconstruction used by ``_dynamic_phase_waveforms`` which
    #     applies ``np.gradient`` (also centred) on the dense grid.
    if scheme not in ("forward", "central"):
        raise ValueError(f"scheme must be 'forward' or 'central', got {scheme!r}")
    coupling_c = omega / (2.0 * delta_theta)  # centred-difference coefficient

    def _add_lin3(coef_self, coef_next, coef_prev, n, n_next, n_prev, sign, offset):
        """Linear inequality with up to three node couplings."""
        full = np.zeros(total_vars)
        full[slot(n)] = sign * coef_self
        full[slot(n_next)] += sign * coef_next
        full[slot(n_prev)] += sign * coef_prev
        c0 = sign * offset

        def fun(X_flat: np.ndarray) -> float:
            return float(full @ X_flat + c0)

        def jac(X_flat: np.ndarray) -> np.ndarray:
            return full

        constraints.append({"type": "ineq", "fun": fun, "jac": jac})

    for n in range(n_grid):
        n_next = (n + 1) % n_grid
        n_prev = (n - 1) % n_grid
        for k in range(n_phases):
            g_kn = G[k, n]
            b_kn = float(bemf_ph[k, n])
            lp_kn = L_proj[k, n]

            if n not in active_nodes:
                constraints.append(_make_lin(coef_self=-g_kn, n=n, sign=1.0, offset=volt_max - b_kn))
                constraints.append(_make_lin(coef_self=g_kn, n=n, sign=1.0, offset=volt_max + b_kn))
                continue

            if scheme == "forward":
                g_kn_next = G[k, n_next]
                b_kn_next = float(bemf_ph[k, n_next])
                lp_kn_next = L_proj[k, n_next]

                # Left edge: U @ x_n + (omega/dtheta) L_proj @ (x_{n+1} - x_n) + bemf
                g_minus = g_kn - coupling * lp_kn
                g_plus = coupling * lp_kn
                constraints.append(
                    _make_lin(
                        coef_self=-g_minus,
                        n=n,
                        sign=1.0,
                        offset=volt_max - b_kn,
                        coef_next=-g_plus,
                        n_next=n_next,
                    )
                )
                constraints.append(
                    _make_lin(
                        coef_self=g_minus,
                        n=n,
                        sign=1.0,
                        offset=volt_max + b_kn,
                        coef_next=g_plus,
                        n_next=n_next,
                    )
                )
                # Right edge of the SAME segment: inductive term unchanged, static
                # contribution evaluated at the right end via U @ x_{n+1}.
                g_minus_r = -coupling * lp_kn_next
                g_plus_r = g_kn_next + coupling * lp_kn_next
                constraints.append(
                    _make_lin(
                        coef_self=-g_minus_r,
                        n=n,
                        sign=1.0,
                        offset=volt_max - b_kn_next,
                        coef_next=-g_plus_r,
                        n_next=n_next,
                    )
                )
                constraints.append(
                    _make_lin(
                        coef_self=g_minus_r,
                        n=n,
                        sign=1.0,
                        offset=volt_max + b_kn_next,
                        coef_next=g_plus_r,
                        n_next=n_next,
                    )
                )
            else:  # scheme == "central"
                # v_k(theta_n) = U @ x_n + (omega/(2 dtheta)) L_proj @ (x_{n+1} - x_{n-1}) + bemf
                a_self = g_kn
                a_next = coupling_c * lp_kn
                a_prev = -coupling_c * lp_kn
                _add_lin3(
                    coef_self=-a_self,
                    coef_next=-a_next,
                    coef_prev=-a_prev,
                    n=n,
                    n_next=n_next,
                    n_prev=n_prev,
                    sign=1.0,
                    offset=volt_max - b_kn,
                )
                _add_lin3(
                    coef_self=a_self,
                    coef_next=a_next,
                    coef_prev=a_prev,
                    n=n,
                    n_next=n_next,
                    n_prev=n_prev,
                    sign=1.0,
                    offset=volt_max + b_kn,
                )

    X0 = X_init.flatten()
    res = minimize(
        objective,
        X0,
        jac=objective_grad,
        method="SLSQP",
        constraints=constraints,
        options=opts,
    )
    return res.x.reshape(n_grid, dim), bool(res.success)


def run_active_set(
    *,
    model: BaseTorqueModel,
    transform: Transform,
    omega: float,
    torq_target: float,
    curr_max: float,
    volt_max: float,
    n_grid: int = 32,
    max_outer_iter: int = 8,
    tol: float = 1e-3,
    opts: dict[str, Any] | None = None,
    rho: float = 0.0,
    scheme: str = "forward",
    iron_weight: float = 0.0,
    excess_weight: float = 0.0,
) -> dict[str, Any]:
    """
    Returns a dict with::

        theta_grid       (n_grid,)
        curr_dq_grid     (n_grid, dim)   -- final trajectory
        r0_curr_dq_grid  (n_grid, dim)   -- R0 (independent) starting point
        active           frozenset       -- active node indices at convergence
        history          list of dicts   -- per outer iteration
        converged        bool
    """
    if opts is None:
        opts = {"disp": False, "ftol": 1e-7, "maxiter": 1500, "eps": 1e-8}

    theta_grid, X0, ok0 = run_independent_per_angle(
        model=model,
        transform=transform,
        omega=omega,
        torq_target=torq_target,
        curr_max=curr_max,
        volt_max=volt_max,
        n_grid=n_grid,
    )
    if not bool(np.all(ok0)):
        return {
            "theta_grid": theta_grid,
            "curr_dq_grid": X0,
            "r0_curr_dq_grid": X0,
            "active": frozenset(),
            "history": [],
            "converged": False,
            "fail_reason": "R0 failed at some nodes",
        }

    X = X0.copy()
    active: frozenset[int] = frozenset()
    history: list[dict[str, Any]] = []

    for outer in range(max_outer_iter):
        r, _ = voltage_residuals(transform, omega, X, volt_max, scheme=scheme)
        violating = frozenset(int(n) for n in np.where(r > tol)[0])
        max_r = float(np.max(r))

        entry = {
            "iter": outer,
            "max_residual_V": max_r,
            "n_violating": len(violating),
            "active_size_before": len(active),
        }

        if not violating:
            history.append(entry | {"action": "converged"})
            return {
                "theta_grid": theta_grid,
                "curr_dq_grid": X,
                "r0_curr_dq_grid": X0,
                "active": active,
                "history": history,
                "converged": True,
            }

        new_active = active | violating
        if new_active == active:
            history.append(entry | {"action": "stuck"})
            return {
                "theta_grid": theta_grid,
                "curr_dq_grid": X,
                "r0_curr_dq_grid": X0,
                "active": active,
                "history": history,
                "converged": False,
            }

        X_new, ok = _solve_joint(
            model,
            transform,
            omega,
            torq_target,
            curr_max,
            volt_max,
            X,
            set(new_active),
            opts,
            rho=rho,
            scheme=scheme,
            iron_weight=iron_weight,
            excess_weight=excess_weight,
        )
        entry["action"] = f"solve(|A|={len(new_active)})"
        entry["solve_success"] = ok
        history.append(entry)
        X = X_new
        active = new_active

    return {
        "theta_grid": theta_grid,
        "curr_dq_grid": X,
        "r0_curr_dq_grid": X0,
        "active": active,
        "history": history,
        "converged": False,
    }


__all__ = ["run_active_set", "voltage_residuals"]
