"""
E15 -- Fourier ansatz vs full position-dependent optimum UNDER FAULT.

Single open-phase fault (phase 0 open) on the 5-phase machine. The four
surviving phases carry the current that five carried before, so the
per-phase current limit binds at lower torque and the loss-optimal
reference shaves the surviving-phase peaks over long arcs. We test
whether that makes the gap between the full optimum and the low-order
Fourier ansatz (prior art) larger than the <1% seen healthy.

Representation. Single fault keeps the dq current i_s fully controllable
(the reduced 4x4 Clarke of the surviving phases is invertible -- no
null-space constraint), with torque A,b and voltage U unchanged. Only
the per-phase map h_k(theta) changes to the surviving-phase rows of the
reduced inverse Clarke (verified bit-for-bit against the healthy
Transform's convention).

At a low speed the voltage limit is slack, so the period decouples and
the full position-dependent optimum equals the per-angle independent
optimum (R0) -- which has unrestricted harmonic content. The Fourier
ansatz restricts i_s(theta) to a chosen harmonic set; H={0} is the
static / prior-art reference.

Run:
    python experiments/fourier/e15_fault_headtohead.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy.optimize import minimize

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from current_setpoints.optimization import ModelAnalytical  # noqa: E402
from current_setpoints.parameters import Flux_IEEEMachine2, IEEEMachine2  # noqa: E402
from current_setpoints.simulation import Transform  # noqa: E402
from current_setpoints.utils.plotting_dynamic import _evaluate_dq_on_grid  # noqa: E402
from dynamic.fourier_optimizer import extract_quadratic_torque, fourier_design, n_basis  # noqa: E402


def _full_clarke_5phase() -> np.ndarray:
    C = np.zeros((5, 5))
    for p in range(5):
        a = p * 2 * np.pi / 5
        C[0, p] = np.cos(a)
        C[1, p] = np.sin(a)
        C[2, p] = np.cos(2 * a)
        C[3, p] = np.sin(2 * a)
        C[4, p] = 0.5
    return C * 2.0 / 5.0


def fault_phase_map(vec_theta: np.ndarray, open_phases: tuple[int, ...]) -> tuple[np.ndarray, tuple[int, ...]]:
    """
    Surviving-phase inverse-Clarke map ``mat_dq_to_ph_fault`` of shape
    ``(n_theta, n_surv, 4)`` such that surviving phase currents at theta are
    ``map[t] @ i_s``. Single-fault: reduced Clarke is square-invertible.
    """
    C_full = _full_clarke_5phase()
    kept = tuple(p for p in range(5) if p not in open_phases)
    C_red = C_full[:4, :][:, list(kept)]
    if C_red.shape[0] != C_red.shape[1]:
        raise NotImplementedError("This experiment handles single-fault (square reduced Clarke) only.")
    T_inv = np.linalg.inv(C_red)
    t = vec_theta
    c1, s1, c3, s3 = np.cos(t), np.sin(t), np.cos(3 * t), np.sin(3 * t)
    R = np.zeros((t.size, 4, 4))
    R[:, 0, 0], R[:, 0, 1], R[:, 1, 0], R[:, 1, 1] = c1, -s1, s1, c1
    R[:, 2, 2], R[:, 2, 3], R[:, 3, 2], R[:, 3, 3] = c3, -s3, s3, c3
    return np.einsum("ij,tjk->tik", T_inv, R), kept


def joule_dense(transform: Transform, theta_grid: np.ndarray, X: np.ndarray) -> float:
    Xd = _evaluate_dq_on_grid(theta_grid, X, transform.vec_theta[:-1])
    return float(np.mean(np.sum(Xd**2, axis=1)))


def per_angle_optimum(model, transform, omega, T, Imax, Vmax, Hf_all, Gf_all, bemf_ph_all, idx_grid):
    """
    Full position-dependent optimum at a current-limited (voltage-slack)
    point: independent per-angle min ||x||^2 s.t. torque + surviving-phase
    current & voltage limits. Returns (theta_grid, X, ok, maxI, maxV).
    """
    dim = transform.dim
    n_grid = idx_grid.size
    X = np.full((n_grid, dim), np.nan)
    seed = np.array([1.0, 1.0, 0.0, 0.0])
    ok = np.zeros(n_grid, dtype=bool)
    for n, i in enumerate(idx_grid):
        Hrows = Hf_all[i]  # (n_surv, dim)
        Grows = Gf_all[i]
        bph = bemf_ph_all[i]
        cons = [{"type": "eq", "fun": (lambda x, _T=T: model.calculate_torque(omega, x) - _T)}]
        for k in range(Hrows.shape[0]):
            cons.append({"type": "ineq", "fun": (lambda x, h=Hrows[k]: Imax - h @ x)})
            cons.append({"type": "ineq", "fun": (lambda x, h=Hrows[k]: Imax + h @ x)})
            cons.append({"type": "ineq", "fun": (lambda x, g=Grows[k], b=bph[k]: Vmax - (g @ x + b))})
            cons.append({"type": "ineq", "fun": (lambda x, g=Grows[k], b=bph[k]: Vmax + (g @ x + b))})
        res = minimize(
            lambda x: float(x @ x), seed, method="SLSQP", constraints=cons, options={"ftol": 1e-10, "maxiter": 800}
        )
        if res.success:
            X[n] = res.x
            seed = res.x
            ok[n] = True
    theta_grid = np.linspace(0.0, 2 * np.pi, n_grid, endpoint=False)
    return theta_grid, X, ok


def fourier_fault(
    model, transform, omega, T, Imax, Vmax, harmonics, Hf_s, Gf_s, bemf_ph_s, Phi, dPhi, ripple_budget=1e-2, warm=None
):
    """Fourier solve under fault with the surviving-phase map at collocation angles."""
    dim = transform.dim
    nb = n_basis(harmonics)
    n_con, n_surv = Hf_s.shape[0], Hf_s.shape[1]
    A_t, b_t, c_t = extract_quadratic_torque(model, omega, dim)
    w = [1.0] + [0.5] * (nb - 1)
    wvec = np.repeat(w, dim)

    def as_C(c):
        return c.reshape(nb, dim)

    def torque_s(c):
        i_s = Phi @ as_C(c)
        return np.einsum("sj,jk,sk->s", i_s, A_t, i_s) + i_s @ b_t + c_t

    cons = [{"type": "eq", "fun": (lambda c: float(np.mean(torque_s(c)) - T))}]
    for s in range(n_con):
        for sign in (+1.0, -1.0):

            def tband(c, p=Phi[s], sg=sign):
                iv = p @ as_C(c)
                return ripple_budget - sg * (float(iv @ A_t @ iv + iv @ b_t + c_t) - T)

            cons.append({"type": "ineq", "fun": tband})
        for k in range(n_surv):
            h = Hf_s[s, k]
            ci = np.outer(Phi[s], h).ravel()
            cons.append({"type": "ineq", "fun": (lambda c, a=ci: Imax - a @ c), "jac": (lambda c, a=ci: -a)})
            cons.append({"type": "ineq", "fun": (lambda c, a=ci: Imax + a @ c), "jac": (lambda c, a=ci: a)})
            gU = transform.mat_curr_dq_to_volt_dq.T @ h
            gL = transform.machine.L_stat.T @ h
            cv = (np.outer(Phi[s], gU) + omega * np.outer(dPhi[s], gL)).ravel()
            b = float(bemf_ph_s[s, k])
            cons.append({"type": "ineq", "fun": (lambda c, a=cv, d=b: Vmax - (a @ c + d)), "jac": (lambda c, a=cv: -a)})
            cons.append({"type": "ineq", "fun": (lambda c, a=cv, d=b: Vmax + (a @ c + d)), "jac": (lambda c, a=cv: a)})

    C0 = np.zeros((nb, dim))
    if warm is not None:
        C0[: min(nb, warm.shape[0])] = warm[:nb]
    else:
        C0[0] = np.array([1.0, 1.0, 0.0, 0.0])
    res = minimize(
        lambda c: float(np.sum(wvec * c * c)),
        C0.ravel(),
        jac=lambda c: 2 * wvec * c,
        method="SLSQP",
        constraints=cons,
        options={"ftol": 1e-10, "maxiter": 4000},
    )
    C = res.x.reshape(nb, dim)
    return C, bool(res.success)


def main() -> None:
    machine = IEEEMachine2()
    machine.set_max_pars(curr_max=30.0, volt_max=13.0, omega_max=1800)
    flux = Flux_IEEEMachine2()
    transform = Transform(machine=machine, flux=flux, add_volt_0=False, n_theta=700)
    model = ModelAnalytical(machine=machine, flux=flux)
    Imax, Vmax = machine.curr_max, machine.volt_max
    rpm = 150.0
    omega = rpm * (np.pi / 30.0) * machine.n_ppairs
    transform._set_omega(omega)
    dim = transform.dim

    # --- single-fault (phase 0 open) maps ---
    Hf_all, kept = fault_phase_map(transform.vec_theta, (0,))  # (n_theta, 4, dim)
    U = transform.mat_curr_dq_to_volt_dq
    flux_volt, _ = transform.flux.get_flux(omega, np.zeros(dim))
    bemf_dq = omega * machine.mat_crossc @ flux_volt
    Gf_all = np.einsum("tik,kj->tij", Hf_all, U)
    bemf_ph_all = np.einsum("tik,k->ti", Hf_all, bemf_dq)
    print(f"single open-phase fault (phase 0); surviving phases {kept}; {rpm:.0f} rpm")

    n_grid = 180
    _, idx_grid = (
        np.linspace(0, 2 * np.pi, n_grid, endpoint=False),
        (
            np.round(
                np.linspace(0, 2 * np.pi, n_grid, endpoint=False) / (2 * np.pi) * (transform.vec_theta.size - 1)
            ).astype(int)
            % (transform.vec_theta.size - 1)
        ),
    )

    # --- find fault current-limited torque ceiling at this speed ---
    ceiling = None
    for T in np.arange(6.5, 2.0, -0.1):
        _, X, ok = per_angle_optimum(
            model, transform, omega, float(T), Imax, Vmax, Hf_all, Gf_all, bemf_ph_all, idx_grid
        )
        if bool(np.all(ok)):
            ceiling = float(T)
            break

    # --- static (constant-dq, prior-art) torque ceiling under fault ---
    # max T(i_s) over a constant dq s.t. surviving-phase peak current <= Imax (voltage slack here).
    Hf_dense = Hf_all  # (n_theta, n_surv, dim)

    def neg_torque(x):
        return -model.calculate_torque(omega, x)

    cons_static = []
    for k in range(Hf_dense.shape[1]):
        cons_static.append({"type": "ineq", "fun": (lambda x, k=k: Imax - np.max(np.abs(Hf_dense[:, k, :] @ x)))})
    res_s = minimize(
        neg_torque,
        np.array([5.0, 5.0, 0.0, 0.0]),
        method="SLSQP",
        constraints=cons_static,
        options={"ftol": 1e-9, "maxiter": 1000},
    )
    static_ceiling = -res_s.fun if res_s.success else float("nan")
    print(
        f"fault torque ceiling @ {rpm:.0f} rpm:  FULL (shaped) ~ {ceiling:.2f} Nm   "
        f"static/prior-art (const dq) ~ {static_ceiling:.2f} Nm   "
        f"-> range extension +{100 * (ceiling - static_ceiling) / static_ceiling:.0f}%"
    )

    for frac, lab in [(0.80, "80% ceiling"), (0.95, "95% ceiling")]:
        T = round(frac * ceiling, 3)
        print(f"\n--- T*={T} Nm ({lab}) ---")
        tg, Xfull, ok = per_angle_optimum(model, transform, omega, T, Imax, Vmax, Hf_all, Gf_all, bemf_ph_all, idx_grid)
        if not bool(np.all(ok)):
            print("  full per-angle optimum infeasible at some node; skipping")
            continue
        Jfull = joule_dense(transform, tg, Xfull)
        # surviving-phase peak current / voltage of full solution at grid nodes
        i_surv = np.einsum("nik,nk->ni", Hf_all[idx_grid], Xfull)  # (n_grid, n_surv)
        v_surv = np.einsum("nik,nk->ni", Gf_all[idx_grid], Xfull) + bemf_ph_all[idx_grid]
        maxI, maxV = np.max(np.abs(i_surv)), np.max(np.abs(v_surv))
        arc = float(np.mean(np.abs(i_surv) >= 0.995 * Imax))
        print(
            f"  FULL optimum: J={Jfull:.3f}  maxIph={maxI:.2f}/{Imax}  maxVph={maxV:.2f}/{Vmax}  arc@Imax={arc * 100:.1f}%"
        )

        # Fourier collocation maps
        n_con = 200
        _, idx_s = (
            np.linspace(0, 2 * np.pi, n_con, endpoint=False),
            (
                np.round(
                    np.linspace(0, 2 * np.pi, n_con, endpoint=False) / (2 * np.pi) * (transform.vec_theta.size - 1)
                ).astype(int)
                % (transform.vec_theta.size - 1)
            ),
        )
        theta_s = transform.vec_theta[idx_s]
        Hf_s = Hf_all[idx_s]
        Gf_s = Gf_all[idx_s]
        bemf_ph_s = bemf_ph_all[idx_s]
        theta_out = np.linspace(0.0, 2 * np.pi, 256, endpoint=False)

        # DC seed = mean of the full optimum (a feasible constant-dq start)
        warm = np.zeros((1, dim))
        warm[0] = Xfull.mean(axis=0)
        for H in [(0,), (0, 9, 11), tuple(range(8)), tuple(range(16))]:
            Phi, dPhi = fourier_design(theta_s, H)
            C, conv = fourier_fault(
                model,
                transform,
                omega,
                T,
                Imax,
                Vmax,
                H,
                Hf_s,
                Gf_s,
                bemf_ph_s,
                Phi,
                dPhi,
                ripple_budget=1e-4,
                warm=warm,
            )
            Xo = fourier_design(theta_out, H)[0] @ C
            Jf = joule_dense(transform, theta_out, Xo)
            # feasibility of this Fourier solution (surviving-phase current peak at collocation)
            i_chk = np.einsum("nik,nk->ni", Hf_all[idx_s], fourier_design(theta_s, H)[0] @ C)
            maxI_f = float(np.max(np.abs(i_chk)))
            gap = 100.0 * (Jf - Jfull) / Jfull
            feas = "OK" if maxI_f <= Imax * 1.002 else f"INFEAS(maxI={maxI_f:.1f})"
            print(
                f"  Fourier H={str(H):16s} J={Jf:8.3f}  gap_vs_full={gap:+.2f}%  maxIph={maxI_f:.2f}  {feas}  conv={conv}"
            )
            warm = C


if __name__ == "__main__":
    main()
