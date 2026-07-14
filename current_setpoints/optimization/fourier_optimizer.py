from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize

from ..models.forward_model import ForwardModel
from .dynamic_optimizer import DynamicOptimizer
from .optimizer import Solution


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
