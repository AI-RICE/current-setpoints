"""
Fourier-domain optimizer for the periodic dynamic problem -- the
prior-art low-order ansatz, recast so it is directly comparable to the
per-angle grid method of ``active_set_optimizer.py``.

Instead of treating the per-node currents ``{x_n}`` as free variables,
the dq trajectory is restricted to a chosen set of *angular harmonics*
in the synchronous frame::

    i_s(theta) = c_0 + sum_{h in H, h>0} [ a_h cos(h theta) + b_h sin(h theta) ]

with one ``(c_0, {a_h, b_h})`` triple per dq component. The decision
variables are the Fourier coefficients ``C`` of shape
``(n_basis, dim)``; ``n_basis = 1 + 2 * #{h in H : h > 0}``.

Why this is the right "prior-art" comparator
---------------------------------------------
For these machines a *constant* dq vector already produces a constant
torque and phase currents made of the back-EMF harmonics (1st + 3rd for
the 5-phase machine here). So ``H = {0}`` is exactly the static MTPA /
low-order Fourier ansatz of Vu et al. / dos Santos Moraes et al. The
grid optimum's advantage lives in higher dq harmonics; the chatter
analysis (E01) located that content at ``h = 10k +/- 1`` (9, 11, 19,
21, ...). Sweeping ``H = {0} -> {0,9,11} -> {0,9,11,19,21} -> ...`` maps
how fast the low-order ansatz closes the gap to the full grid, and which
augmentation is the cheap win.

Formulation details
--------------------
* **Objective** is the period-mean Joule loss
  ``(1/2pi) int ||i_s||^2 dtheta``. By Parseval this is
  ``||c_0||^2 + 1/2 sum_{h>0}(||a_h||^2 + ||b_h||^2)`` -- exact, with an
  analytic gradient.
* **Torque** is held by a *mean* equality ``mean_theta T(i_s) = T*``
  plus a pointwise ripple band ``|T(i_s(theta)) - T*| <= ripple_budget``.
  (A strict pointwise *equality* at every sample would exceed SLSQP's
  "more equalities than variables" limit for a truncated basis; the band
  is the standard low-order-ansatz treatment and exposes the
  Joule/ripple trade-off.) The machine is salient, so torque is quadratic;
  we recover ``T(i) = i^T A i + b^T i + c`` once by least squares from
  ``model.calculate_torque`` and use it analytically (value + gradient),
  which keeps every constraint Jacobian closed-form and the solve fast.
* **Current / voltage** peak limits are enforced two-sided at the sample
  angles and are *linear* in ``C`` (so we hand SLSQP analytic
  Jacobians). The voltage uses the **exact** inductive term
  ``omega L_s di_s/dtheta`` -- the Fourier derivative is analytic, so
  unlike the grid solver there is no forward/central-difference choice.

The sample angles are snapped to ``transform.vec_theta`` so the phase
projection ``h_k(theta)`` matches the grid solver bit-for-bit.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize

from current_setpoints.optimization.models import BaseTorqueModel
from current_setpoints.simulation import Transform
from dynamic.active_set_optimizer import _grid_indices, _phase_basis
from dynamic.independent_optimizer import run_independent_per_angle


def _positive_harmonics(harmonics: tuple[int, ...]) -> list[int]:
    return [int(h) for h in sorted(set(harmonics)) if int(h) > 0]


def n_basis(harmonics: tuple[int, ...]) -> int:
    """Number of Fourier basis functions (1 for DC + 2 per positive harmonic)."""
    has_dc = 0 in set(int(h) for h in harmonics)
    return (1 if has_dc else 0) + 2 * len(_positive_harmonics(harmonics))


def fourier_design(theta: np.ndarray, harmonics: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """
    Design matrices ``(Phi, dPhi)`` of shape ``(len(theta), n_basis)``.

    Columns are ordered ``[1, cos(h1 t), sin(h1 t), cos(h2 t), sin(h2 t), ...]``
    (the leading constant column is present iff ``0 in harmonics``).
    ``dPhi`` is the elementwise derivative ``d/dtheta``.
    """
    theta = np.asarray(theta, dtype=float)
    cols: list[np.ndarray] = []
    dcols: list[np.ndarray] = []
    if 0 in set(int(h) for h in harmonics):
        cols.append(np.ones_like(theta))
        dcols.append(np.zeros_like(theta))
    for h in _positive_harmonics(harmonics):
        cols.append(np.cos(h * theta))
        dcols.append(-h * np.sin(h * theta))
        cols.append(np.sin(h * theta))
        dcols.append(h * np.cos(h * theta))
    return np.stack(cols, axis=1), np.stack(dcols, axis=1)


def _parseval_weights(harmonics: tuple[int, ...]) -> np.ndarray:
    """Per-basis weight in the period-mean of the square: 1 for DC, 1/2 for cos/sin."""
    w: list[float] = []
    if 0 in set(int(h) for h in harmonics):
        w.append(1.0)
    for _ in _positive_harmonics(harmonics):
        w.extend([0.5, 0.5])
    return np.asarray(w, dtype=float)


def reconstruct(coeffs: np.ndarray, theta: np.ndarray, harmonics: tuple[int, ...]) -> np.ndarray:
    """Evaluate the dq trajectory ``i_s(theta)`` of shape ``(len(theta), dim)``."""
    Phi, _ = fourier_design(theta, harmonics)
    return Phi @ coeffs


def extract_quadratic_torque(
    model: BaseTorqueModel, omega: float, dim: int, scale: float = 8.0, n_samples: int = 300
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Recover ``(A, b, c)`` such that ``T(i) = i^T A i + b^T i + c`` exactly,
    by least-squares from ``model.calculate_torque`` (torque is quadratic in
    the dq currents for these machines). ``A`` is returned symmetric.
    """
    rng = np.random.default_rng(0)
    Xs = rng.normal(scale=scale, size=(n_samples, dim))
    idx = [(i, j) for i in range(dim) for j in range(i, dim)]
    feats = np.array([[x[i] * x[j] for (i, j) in idx] + list(x) + [1.0] for x in Xs])
    y = np.array([model.calculate_torque(omega, x) for x in Xs])
    coef = np.linalg.lstsq(feats, y, rcond=None)[0]
    A = np.zeros((dim, dim))
    for q, (i, j) in enumerate(idx):
        if i == j:
            A[i, i] = coef[q]
        else:
            A[i, j] = A[j, i] = coef[q] / 2.0
    b = coef[len(idx) : len(idx) + dim]
    c = float(coef[-1])
    return A, b, c


def run_fourier(
    *,
    model: BaseTorqueModel,
    transform: Transform,
    omega: float,
    torq_target: float,
    curr_max: float,
    volt_max: float,
    harmonics: tuple[int, ...] = (0, 9, 11),
    n_con: int = 180,
    n_out: int = 256,
    ripple_budget: float = 1e-2,
    warm_coeffs: np.ndarray | None = None,
    opts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Solve the periodic problem restricted to the dq angular harmonics in
    ``harmonics`` (must include 0). Mirrors ``run_active_set``'s call
    signature and operating-point convention.

    Returns a dict with::

        harmonics        tuple
        coeffs           (n_basis, dim)   -- optimal Fourier coefficients
        theta_grid       (n_out,)         -- output sampling grid
        curr_dq_grid     (n_out, dim)     -- trajectory on theta_grid
        joule            float            -- period-mean Joule (Parseval)
        torque_ripple    float            -- max |T(theta) - T*| on n_con grid
        ripple_budget    float            -- band used for the torque constraint
        converged        bool
        n_basis          int
    """
    if 0 not in set(int(h) for h in harmonics):
        raise ValueError("harmonics must include the DC term 0.")
    if opts is None:
        opts = {"disp": False, "ftol": 1e-9, "maxiter": 2000, "eps": 1e-8}

    transform._set_omega(omega)
    dim = transform.dim
    n_phases = transform.n_phases
    nb = n_basis(harmonics)

    # Sample angles snapped to vec_theta so h_k(theta) matches the grid solver.
    theta_s, idx_s = _grid_indices(transform, n_con)
    theta_s = transform.vec_theta[idx_s]
    H = _phase_basis(transform, idx_s)  # (n_phases, n_con, dim)

    Phi, dPhi = fourier_design(theta_s, harmonics)  # (n_con, nb)

    U = transform.mat_curr_dq_to_volt_dq
    L = transform.machine.L_stat
    flux_volt, _ = transform.flux.get_flux(omega, np.zeros(dim))
    bemf_dq = omega * transform.machine.mat_crossc @ flux_volt
    bemf_ph = np.einsum("knj,j->kn", H, bemf_dq)  # (n_phases, n_con)

    # analytic quadratic torque T(i) = i^T A i + b^T i + c  (grad = 2 A i + b)
    A_t, b_t, c_t = extract_quadratic_torque(model, omega, dim)

    total_vars = nb * dim
    wvec = np.repeat(_parseval_weights(harmonics), dim)  # length total_vars

    # ----- objective: period-mean Joule via Parseval (exact) -----
    def objective(c: np.ndarray) -> float:
        return float(np.sum(wvec * c * c))

    def objective_grad(c: np.ndarray) -> np.ndarray:
        return 2.0 * wvec * c

    def as_C(c: np.ndarray) -> np.ndarray:
        return c.reshape(nb, dim)

    def torque_at(c: np.ndarray) -> np.ndarray:
        """Torque at every sample angle, shape (n_con,)."""
        i_s = Phi @ as_C(c)  # (n_con, dim)
        return np.einsum("sj,jk,sk->s", i_s, A_t, i_s) + i_s @ b_t + c_t

    def torque_grad_rows(c: np.ndarray) -> np.ndarray:
        """d T(theta_s)/d c, shape (n_con, total_vars)."""
        i_s = Phi @ as_C(c)
        g_i = 2.0 * (i_s @ A_t) + b_t[None, :]  # (n_con, dim)
        # dT_s/dC[p,d] = Phi[s,p] * g_i[s,d]
        return (Phi[:, :, None] * g_i[:, None, :]).reshape(n_con, total_vars)

    constraints: list[dict[str, Any]] = []

    # ----- torque: mean equality + pointwise ripple band -----
    def eq_mean_torque(c: np.ndarray) -> float:
        return float(np.mean(torque_at(c)) - torq_target)

    def eq_mean_torque_jac(c: np.ndarray) -> np.ndarray:
        return torque_grad_rows(c).mean(axis=0)

    constraints.append({"type": "eq", "fun": eq_mean_torque, "jac": eq_mean_torque_jac})

    # ripple band, per sample: ripple_budget -/+ (T_s - T*) >= 0.
    # Kept as individual scalar constraints (not one vector constraint): with
    # an infeasible static seed SLSQP recovers reliably from the scalar form.
    def _torque_one(s: int, sign: float):
        phi_s = Phi[s]

        def fun(c: np.ndarray) -> float:
            i_sv = phi_s @ as_C(c)
            return ripple_budget - sign * (float(i_sv @ A_t @ i_sv + i_sv @ b_t + c_t) - torq_target)

        def jac(c: np.ndarray) -> np.ndarray:
            i_sv = phi_s @ as_C(c)
            g_i = 2.0 * (A_t @ i_sv) + b_t
            return -sign * np.outer(phi_s, g_i).ravel()

        return {"type": "ineq", "fun": fun, "jac": jac}

    for s in range(n_con):
        constraints.append(_torque_one(s, +1.0))
        constraints.append(_torque_one(s, -1.0))

    # ----- current / voltage limits, two-sided, linear in C with analytic jac -----
    # A linear inequality ``coef . c + offset >= 0``.
    def _lin(coef: np.ndarray, offset: float) -> dict[str, Any]:
        def fun(c: np.ndarray) -> float:
            return float(coef @ c + offset)

        def jac(c: np.ndarray) -> np.ndarray:
            return coef

        return {"type": "ineq", "fun": fun, "jac": jac}

    # h_k . i_s = sum_{b,d} h_k[d] Phi[s,b] C[b,d] = outer(Phi[s], h_k).ravel() . c
    for s in range(n_con):
        for k in range(n_phases):
            h_kn = H[k, s]  # (dim,)
            coef_i = np.outer(Phi[s], h_kn).ravel()  # d(h.i)/dc
            constraints.append(_lin(-coef_i, curr_max))  # curr_max - h.i >= 0
            constraints.append(_lin(coef_i, curr_max))  # curr_max + h.i >= 0

    # v_k(theta_s) = (U^T h_k) . i_s + omega (L^T h_k) . di_s/dtheta + h_k . bemf
    #             = sum_{b,d}[ gU[d] Phi[s,b] + omega gL[d] dPhi[s,b] ] C[b,d] + bemf_ph
    for s in range(n_con):
        for k in range(n_phases):
            h_kn = H[k, s]
            gU = U.T @ h_kn  # (dim,)
            gL = L.T @ h_kn  # (dim,)
            coef_v = (np.outer(Phi[s], gU) + omega * np.outer(dPhi[s], gL)).ravel()
            b_kn = float(bemf_ph[k, s])
            constraints.append(_lin(-coef_v, volt_max - b_kn))  # Vmax - v >= 0
            constraints.append(_lin(coef_v, volt_max + b_kn))  # Vmax + v >= 0

    # ----- warm start -----
    # A good DC seed matters: the static-feasible constant-dq solution is the
    # natural starting point. If no warm_coeffs given and the basis carries AC
    # harmonics, pre-solve the DC-only ({0}) problem first for that seed.
    C0 = np.zeros((nb, dim))
    if warm_coeffs is not None:
        C0[: min(nb, warm_coeffs.shape[0])] = warm_coeffs[:nb]
    else:
        if nb > dim:  # has AC harmonics -> seed DC from the static solve
            dc = run_fourier(
                model=model,
                transform=transform,
                omega=omega,
                torq_target=torq_target,
                curr_max=curr_max,
                volt_max=volt_max,
                harmonics=(0,),
                n_con=n_con,
                n_out=n_out,
                ripple_budget=ripple_budget,
                opts=opts,
            )
            C0[0] = dc["coeffs"][0]
        else:
            _, X_r0, ok_r0 = run_independent_per_angle(
                model=model,
                transform=transform,
                omega=omega,
                torq_target=torq_target,
                curr_max=curr_max,
                volt_max=volt_max,
                n_grid=32,
            )
            if bool(np.any(ok_r0)):
                C0[0] = np.nanmean(X_r0[ok_r0], axis=0)

    res = minimize(
        objective,
        C0.ravel(),
        jac=objective_grad,
        method="SLSQP",
        constraints=constraints,
        options=opts,
    )
    C = res.x.reshape(nb, dim)
    torque_ripple = float(np.max(np.abs(torque_at(res.x) - torq_target)))

    theta_out = np.linspace(0.0, 2 * np.pi, n_out, endpoint=False)
    return {
        "harmonics": tuple(sorted(set(int(h) for h in harmonics))),
        "coeffs": C,
        "theta_grid": theta_out,
        "curr_dq_grid": reconstruct(C, theta_out, harmonics),
        "joule": objective(res.x),
        "torque_ripple": torque_ripple,
        "ripple_budget": ripple_budget,
        "converged": bool(res.success),
        "n_basis": nb,
    }


__all__ = ["run_fourier", "reconstruct", "fourier_design", "n_basis"]
