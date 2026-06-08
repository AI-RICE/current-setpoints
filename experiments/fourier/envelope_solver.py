"""
Fourier-on-fault-map envelope solver -- shared machinery for the
single-open-phase fault head-to-head experiments in this folder.

Provides three arms, all parametrised by the SAME solver so the only
differences are the constraint set (not the trajectory representation):

  * Static   -- harmonics {0} (constant dq) + per-phase current peak +
                per-phase rms + per-phase voltage (steady-state for
                constant dq, no di/dtheta term).
  * Yepes    -- harmonics {0..N_F} (free shaping) + current peak + rms,
                NO voltage limit (matches Yepes 2024 NSBE-FRML structure).
  * Dynamic  -- harmonics {0..N_F} (free shaping) + current peak + rms +
                per-phase voltage incl. the EXACT inductive
                omega*L*di_s/dtheta term (this work).

Note: Fall (Eqs 16-19 verbatim) is a *different* parametrisation
(2-DOF, closed-form sinusoidal surviving currents) and is implemented
inline in the e17 script -- see baselines/Fall/ for the standalone
faithful Fall replication.

Voltage convention (unified, validated -- see dynamic/phase_voltage.py):
per-phase voltage uses the inverse-Park dq->phase map, the machine's
native, fault-independent voltage transform (the SAME one the validated
``Transform.get_volt_ph`` uses). Current under fault uses the
reduced-Clarke map (``fault_phase_map``) so the open phase carries no
current. These two maps genuinely differ and the asymmetry is physical.
An earlier version reused the reduced-Clarke *current* map on v_dq, which
over-counted surviving-phase voltage once omega*L*di/dtheta mattered and
spuriously collapsed the Dynamic envelope at high speed. The voltage
linear maps (gU, gL, bV) are now supplied by the caller via
``phase_voltage.voltage_linear_maps`` and passed through ``maps``.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy.optimize import minimize

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dynamic.fourier_optimizer import extract_quadratic_torque, n_basis  # noqa: E402

H_FREE = tuple(range(11))  # {0..10} free-shaping basis for Yepes / Dynamic
N_CON = 90  # collocation grid size

ARMS = ("Static", "Yepes", "Dynamic")


def full_clarke_5phase() -> np.ndarray:
    """5x5 amplitude-invariant Clarke (rows: alpha1, beta1, alpha3, beta3, zero)."""
    C = np.zeros((5, 5))
    for p in range(5):
        a = p * 2 * np.pi / 5
        C[0, p], C[1, p], C[2, p], C[3, p], C[4, p] = np.cos(a), np.sin(a), np.cos(2 * a), np.sin(2 * a), 0.5
    return C * 2.0 / 5.0


def fault_phase_map(vec_theta: np.ndarray, open_phases: tuple[int, ...]) -> np.ndarray:
    """
    Reduced inverse Clarke for the surviving phases under fault.
    Returns array of shape (n_theta, n_surv, dim) mapping dq -> surviving
    phase currents at every angle.
    """
    C_full = full_clarke_5phase()
    kept = [p for p in range(5) if p not in open_phases]
    C_red = C_full[:4, :][:, kept]
    T_inv = np.linalg.inv(C_red)
    t = vec_theta
    c1, s1, c3, s3 = np.cos(t), np.sin(t), np.cos(3 * t), np.sin(3 * t)
    R = np.zeros((t.size, 4, 4))
    R[:, 0, 0], R[:, 0, 1], R[:, 1, 0], R[:, 1, 1] = c1, -s1, s1, c1
    R[:, 2, 2], R[:, 2, 3], R[:, 3, 2], R[:, 3, 3] = c3, -s3, s3, c3
    return np.einsum("ij,tjk->tik", T_inv, R)


def solve_fault(
    model,
    transform,
    omega,
    T,
    harmonics,
    maps,
    Imax,
    Vmax,
    use_voltage,
    rms_max,
    ripple_budget=1e-2,
    warm=None,
):
    """
    Solve the min-loss Fourier-on-fault-map problem at torque T.
    Returns (C, converged, maxI, maxV, max_rms_per_phase).

    ``maps`` = (Hf_s, gU, gL, bV, Phi, dPhi) where:
      Hf_s (n_con, n_surv, dim) -- reduced-Clarke CURRENT map (i_0 = 0).
      gU, gL (n_con, n_surv, dim), bV (n_con, n_surv) -- inverse-Park
        VOLTAGE linear maps from phase_voltage.voltage_linear_maps:
        v_phase = gU . i_dq + omega * gL . di_dq + bV.
    """
    Hf_s, gU, gL, bV, Phi, dPhi = maps
    dim = transform.dim
    nb = n_basis(harmonics)
    n_con, n_surv = Hf_s.shape[0], Hf_s.shape[1]
    nvars = nb * dim
    A_t, b_t, c_t = extract_quadratic_torque(model, omega, dim)
    wvec = np.repeat([1.0] + [0.5] * (nb - 1), dim)

    def as_C(c):
        return c.reshape(nb, dim)

    def torque_s(c):
        i_s = Phi @ as_C(c)
        return np.einsum("sj,jk,sk->s", i_s, A_t, i_s) + i_s @ b_t + c_t

    def torque_grad_rows(c):
        i_s = Phi @ as_C(c)
        g_i = 2.0 * (i_s @ A_t) + b_t[None, :]
        return (Phi[:, :, None] * g_i[:, None, :]).reshape(n_con, nvars)

    Ci = (Phi[:, None, :, None] * Hf_s[:, :, None, :]).reshape(n_con * n_surv, nvars)

    cons = [
        {
            "type": "eq",
            "fun": (lambda c: float(np.mean(torque_s(c)) - T)),
            "jac": (lambda c: torque_grad_rows(c).mean(axis=0)),
        },
        {"type": "ineq", "fun": (lambda c: ripple_budget - (torque_s(c) - T)), "jac": (lambda c: -torque_grad_rows(c))},
        {"type": "ineq", "fun": (lambda c: ripple_budget + (torque_s(c) - T)), "jac": (lambda c: torque_grad_rows(c))},
        {"type": "ineq", "fun": (lambda c: Imax - Ci @ c), "jac": (lambda c: -Ci)},
        {"type": "ineq", "fun": (lambda c: Imax + Ci @ c), "jac": (lambda c: Ci)},
    ]
    if use_voltage:
        Cv = (Phi[:, None, :, None] * gU[:, :, None, :] + omega * dPhi[:, None, :, None] * gL[:, :, None, :]).reshape(
            n_con * n_surv, nvars
        )
        bvec = bV.reshape(n_con * n_surv)
        cons += [
            {"type": "ineq", "fun": (lambda c: Vmax - (Cv @ c + bvec)), "jac": (lambda c: -Cv)},
            {"type": "ineq", "fun": (lambda c: Vmax + (Cv @ c + bvec)), "jac": (lambda c: Cv)},
        ]
    if rms_max is not None:

        def rms_block(c):
            iph = (Ci @ c).reshape(n_con, n_surv)
            return rms_max**2 - np.mean(iph**2, axis=0)

        def rms_block_jac(c):
            iph = (Ci @ c).reshape(n_con, n_surv)
            Ci3 = Ci.reshape(n_con, n_surv, nvars)
            return -(2.0 / n_con) * np.einsum("sk,skv->kv", iph, Ci3)

        cons.append({"type": "ineq", "fun": rms_block, "jac": rms_block_jac})

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
        options={"ftol": 1e-7, "maxiter": 800},
    )
    C = res.x.reshape(nb, dim)
    i_s = Phi @ C
    i_ph = np.einsum("skj,sj->sk", Hf_s, i_s)
    maxI = float(np.max(np.abs(i_ph)))
    rms = float(np.max(np.sqrt(np.mean(i_ph**2, axis=0))))
    if use_voltage:
        di = dPhi @ C
        v_ph = np.einsum("skl,sl->sk", gU, i_s) + omega * np.einsum("skl,sl->sk", gL, di) + bV
        maxV = float(np.max(np.abs(v_ph)))
    else:
        maxV = 0.0
    return C, bool(res.success), maxI, maxV, rms


def feasible(model, transform, omega, T, arm, maps, Imax, Vmax, rms_max, warm):
    """Is torque T feasible under this arm's constraint set?"""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
    harmonics = (0,) if arm == "Static" else H_FREE
    use_voltage = arm in ("Static", "Dynamic")
    C, conv, maxI, maxV, rms = solve_fault(
        model,
        transform,
        omega,
        T,
        harmonics,
        maps,
        Imax,
        Vmax,
        use_voltage,
        rms_max,
        warm=warm,
    )
    ok = conv and maxI <= Imax * 1.01
    if use_voltage:
        ok = ok and maxV <= Vmax * 1.01
    if rms_max is not None:
        ok = ok and rms <= rms_max * 1.01
    return ok, C


def max_torque(model, transform, omega, arm, maps, Imax, Vmax, rms_max, T_hi=7.0):
    """Bisect the max ripple-free torque feasible for this arm."""
    lo, hi = 0.0, T_hi
    warm = None
    ok0, C0 = feasible(model, transform, omega, 0.5, arm, maps, Imax, Vmax, rms_max, None)
    if ok0:
        warm = C0
    for _ in range(8):
        mid = 0.5 * (lo + hi)
        ok, C = feasible(model, transform, omega, mid, arm, maps, Imax, Vmax, rms_max, warm)
        if ok:
            lo = mid
            warm = C
        else:
            hi = mid
    return lo


__all__ = [
    "ARMS",
    "H_FREE",
    "N_CON",
    "fault_phase_map",
    "feasible",
    "full_clarke_5phase",
    "max_torque",
    "solve_fault",
]
