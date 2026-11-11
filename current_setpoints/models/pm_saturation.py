"""Saturating PM flux model for a five-phase PMSM (planes h = 1, 3), physics-based.

Inverse (current-of-flux) form derived from ONE magnetic energy W(psi), so the
differential inductance L = (d^2 W / d psi^2)^-1 is symmetric (reciprocity) by
construction, and the shape between and beyond measured points is set by the
model, not by the samples. In the magnet-aligned frame (x1, y1, x3, y3) =
(psi_d1, psi_q1, psi_d3, psi_q3):

  i = grad W(psi) - (i_f1, 0, i_f3, 0)              magnets = current sources behind the iron
  W = a_d0/2 x1^2 + a_d0 |x1|^(S+2) / ((S+2) Psi_d^S)                 d-axis saturation
    + a_q0/2 y1^2 + a_q0 |y1|^(T+2) / ((T+2) Psi_q^T)                 q-axis saturation
    + a_dq |x1|^(U+2) |y1|^(V+2) / ((U+2)(V+2))                        cross saturation
    + g(r)/2 [b_d (x3 - o_d)^2 + b_q (y3 - o_q)^2]                     3rd plane
    + c_d x1 x3 + c_q y1 y3                                            linear plane coupling
  g(r) = 1 + k_s r^p,  r = |psi_1|          3rd-plane inductance falls as the iron saturates
  o = K h(r) (psi_1/|psi_1|)^3 + (k_xx x1^2 + k_yy y1^2, k_xy x1 y1)   saturation harmonic:
      round-rotor part (h = z/(1+z), z = (r/Psi_o)^q) + salient-rotor quadratic part

The plane-1 part is the algebraic saturation model of Hinkkanen et al., IEEE
Trans. Ind. Appl. 53(3), 2017; the 3rd-plane terms mirror the traction-IM model
(fluxfit_c). W is even in (y1, y3) -- the machine is mirror-symmetric about d --
so the generating quadrant follows. The measured dq frame may be offset from the
magnet axis by theta0 in plane 1 and by theta3 in plane 3 (= 3 theta0 for a pure
encoder offset; free, because the 3rd-harmonic magnet axis can sit apart); all
public methods use the measured (controller) frame.

As a FluxModel composed into PMSMDrive: flux() returns psi(i) - L(i) i and
inductance() the symmetric differential L(i), so PMSMDrive's torque and voltage
equal (m p/2) i.J.psi(i) and R i + omega J psi(i) exactly.
"""

from __future__ import annotations

import json

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit

from .flux_map_drive import _DATA_DIR, _PMSM5_CAMPAIGNS, _PMSM5_DEFAULT, _pmsm5_setup
from .machines import FluxModel, PMSMDrive, PMSMParams, _build_cross_coupling

# name, initial, lower, upper, log-parametrised
SPEC = [
    ("a_d0", 222.0, 50.0, 1000.0, True), ("S", 4.0, 0.5, 15.0, False), ("Psi_d", 0.3, 0.05, 2.0, True),
    ("a_q0", 95.0, 20.0, 1000.0, True), ("T", 2.0, 0.5, 15.0, False), ("Psi_q", 0.3, 0.02, 2.0, True),
    ("a_dq", 1e3, 1e-2, 1e8, True), ("U", 1.0, 0.0, 6.0, False), ("V", 0.5, 0.0, 6.0, False),
    ("i_f1", 58.0, 1.0, 500.0, True),
    ("b_d", 400.0, 50.0, 3000.0, True), ("b_q", 400.0, 50.0, 3000.0, True),
    ("k_s", 1.0, 1e-4, 1e4, True), ("p", 2.0, 0.5, 15.0, False),
    ("i_f3", -9.6, -200.0, 200.0, False),
    ("K", 0.0, -0.1, 0.1, False), ("Psi_o", 0.2, 0.03, 1.0, True), ("q", 6.0, 1.0, 20.0, False),
    ("c_d", 0.0, -2000.0, 2000.0, False), ("c_q", 0.0, -2000.0, 2000.0, False),
    ("theta0", 0.037, -0.2, 0.2, False),
    ("k_xx", 0.0, -5.0, 5.0, False), ("k_yy", 0.0, -5.0, 5.0, False), ("k_xy", 0.0, -5.0, 5.0, False),
    ("theta3", 0.11, -0.6, 0.6, False),
]
NAMES = [s[0] for s in SPEC]
_LOG = np.array([s[4] for s in SPEC])
_EPS = 1e-9


def to_theta(p: dict) -> np.ndarray:
    return np.array([np.log(p[n]) if lg else p[n] for (n, _, _, _, lg) in SPEC])


def to_params(th: np.ndarray) -> dict:
    return {n: float(np.exp(v) if lg else v) for (n, _, _, _, lg), v in zip(SPEC, th)}


P0 = {s[0]: s[1] for s in SPEC}
LO = to_theta({s[0]: s[2] for s in SPEC})
HI = to_theta({s[0]: s[3] for s in SPEC})


def _frame(theta0: float, theta3: float) -> np.ndarray:
    """Q maps measured-frame dq vectors to the magnet-aligned frame."""
    def rot(a):
        return np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    Q = np.zeros((4, 4))
    Q[:2, :2], Q[2:, 2:] = rot(theta0), rot(theta3)
    return Q


def grad_w(p: dict, psi: np.ndarray) -> np.ndarray:
    """grad W in the magnet-aligned frame; psi (n, 4) -> (n, 4)."""
    x1, y1, x3, y3 = psi.T
    ax, ay = np.abs(x1) + _EPS, np.abs(y1) + _EPS
    r = np.sqrt(x1 * x1 + y1 * y1) + _EPS
    a_dd = p["a_d0"] / p["Psi_d"] ** p["S"]
    a_qq = p["a_q0"] / p["Psi_q"] ** p["T"]
    # plane 1 (Hinkkanen form)
    g_x1 = p["a_d0"] * x1 + a_dd * ax ** p["S"] * x1 + p["a_dq"] / (p["V"] + 2) * ax ** p["U"] * x1 * ay ** (p["V"] + 2)
    g_y1 = p["a_q0"] * y1 + a_qq * ay ** p["T"] * y1 + p["a_dq"] / (p["U"] + 2) * ax ** (p["U"] + 2) * ay ** p["V"] * y1
    # saturation harmonic o = K m(r) (P_d, P_q), m = h/r^3
    h = expit(p["q"] * np.log(r / p["Psi_o"]))  # = z/(1+z), z = (r/Psi_o)^q, overflow-safe
    dh = p["q"] / r * h * (1 - h)
    m = h / r**3
    dm = dh / r**3 - 3 * h / r**4
    Pd, Pq = x1**3 - 3 * x1 * y1**2, 3 * x1**2 * y1 - y1**3
    K, kxx, kyy, kxy = p["K"], p["k_xx"], p["k_yy"], p["k_xy"]
    o_d = K * m * Pd + kxx * x1**2 + kyy * y1**2
    o_q = K * m * Pq + kxy * x1 * y1
    do_d_x = K * (dm * x1 / r * Pd + m * (3 * x1**2 - 3 * y1**2)) + 2 * kxx * x1
    do_d_y = K * (dm * y1 / r * Pd - m * 6 * x1 * y1) + 2 * kyy * y1
    do_q_x = K * (dm * x1 / r * Pq + m * 6 * x1 * y1) + kxy * y1
    do_q_y = K * (dm * y1 / r * Pq + m * (3 * x1**2 - 3 * y1**2)) + kxy * x1
    # 3rd plane with saturation-dependent inverse inductance g(r)
    g = 1 + p["k_s"] * r ** p["p"]
    dg_r = p["k_s"] * p["p"] * r ** (p["p"] - 1)
    u, v = x3 - o_d, y3 - o_q
    e3 = p["b_d"] * u * u + p["b_q"] * v * v
    gu, gv = g * p["b_d"] * u, g * p["b_q"] * v
    g_x1 = g_x1 + 0.5 * dg_r * x1 / r * e3 - (gu * do_d_x + gv * do_q_x) + p["c_d"] * x3
    g_y1 = g_y1 + 0.5 * dg_r * y1 / r * e3 - (gu * do_d_y + gv * do_q_y) + p["c_q"] * y3
    g_x3 = gu + p["c_d"] * x1
    g_y3 = gv + p["c_q"] * y1
    return np.stack([g_x1, g_y1, g_x3, g_y3], axis=1)


def current_true(p: dict, psi: np.ndarray) -> np.ndarray:
    return grad_w(p, psi) - np.array([p["i_f1"], 0.0, p["i_f3"], 0.0])


def hessian_w(p: dict, psi: np.ndarray, h: float = 1e-7) -> np.ndarray:
    """d^2 W / d psi^2 by central differences of the analytic gradient (n, 4, 4), symmetrised."""
    H = np.empty((len(psi), 4, 4))
    for j in range(4):
        e = np.zeros(4)
        e[j] = h
        H[:, :, j] = (grad_w(p, psi + e) - grad_w(p, psi - e)) / (2 * h)
    return 0.5 * (H + np.transpose(H, (0, 2, 1)))


def solve_flux_true(p: dict, i_t: np.ndarray, guess: np.ndarray | None = None, tol: float = 1e-9,
                    maxit: int = 40, max_step: float = 0.05) -> np.ndarray:
    """psi with current_true(psi) = i_t, damped Newton, vectorised over rows; steps
    capped at max_step Wb and a near-singular Hessian regularised, so far-out
    queries (an optimizer exploring) cannot diverge."""
    i_t = np.atleast_2d(i_t)
    if guess is None:
        psi = np.array([0.2, 0.0, -0.02, 0.0]) + i_t / np.array([p["a_d0"], p["a_q0"], p["b_d"], p["b_q"]])
    else:
        psi = np.array(guess, float).reshape(i_t.shape)
    F = current_true(p, psi) - i_t
    for _ in range(maxit):
        err = np.abs(F).max(axis=1)
        if err.max() < tol:
            break
        H = hessian_w(p, psi)
        H = H + 1e-9 * np.abs(H).max(axis=(1, 2))[:, None, None] * np.eye(4)
        try:
            step = np.linalg.solve(H, -F[:, :, None])[:, :, 0]
        except np.linalg.LinAlgError:
            step = np.stack([np.linalg.lstsq(Hn, -Fn, rcond=None)[0] for Hn, Fn in zip(H, F)])
        big = np.abs(step).max(axis=1, keepdims=True)
        step = np.where(big > max_step, step * max_step / np.maximum(big, 1e-300), step)
        lam = np.ones(len(psi))
        for _ in range(20):
            cand = psi + lam[:, None] * step
            Fc = current_true(p, cand) - i_t
            bad = np.abs(Fc).max(axis=1) > err
            if not bad.any():
                break
            lam = np.where(bad, 0.5 * lam, lam)
        psi, F = cand, Fc
    return psi


class SaturatingPMFlux(FluxModel):
    """Fitted physics model as a FluxModel (measured frame); see module docstring."""

    def __init__(self, params: dict) -> None:
        self.p = {"k_xx": 0.0, "k_yy": 0.0, "k_xy": 0.0, "theta3": 3 * params["theta0"], **params}
        self.cross_coupling = _build_cross_coupling(2)
        self._Q = _frame(self.p["theta0"], self.p["theta3"])
        self._cache: tuple[tuple[float, ...], np.ndarray, np.ndarray] | None = None

    def current(self, psi: np.ndarray) -> np.ndarray:
        """Inverse model: currents (n, 4) for fluxes (n, 4), measured frame."""
        return current_true(self.p, np.atleast_2d(psi) @ self._Q.T) @ self._Q

    def total_flux(self, curr_dq: np.ndarray, guess: np.ndarray | None = None) -> np.ndarray:
        """Forward model: fluxes (n, 4) for currents (n, 4), measured frame."""
        i_t = np.atleast_2d(curr_dq) @ self._Q.T
        g = None if guess is None else np.atleast_2d(guess) @ self._Q.T
        return solve_flux_true(self.p, i_t, g) @ self._Q

    def _eval(self, curr_dq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        key = tuple(float(v) for v in curr_dq)
        if self._cache is not None and self._cache[0] == key:
            return self._cache[1], self._cache[2]
        warm = None if self._cache is None else self._cache[1] @ self._Q.T
        psi_t = solve_flux_true(self.p, np.asarray(curr_dq, float)[None, :] @ self._Q.T, warm)
        L = self._Q.T @ np.linalg.inv(hessian_w(self.p, psi_t)[0]) @ self._Q
        psi = (psi_t @ self._Q)[0]
        self._cache = (key, psi, 0.5 * (L + L.T))
        return psi, self._cache[2]

    def flux(self, omega: float, curr_dq: np.ndarray) -> np.ndarray:
        psi, L = self._eval(curr_dq)
        return psi - L @ np.asarray(curr_dq, float)

    def inductance(self, omega: float, curr_dq: np.ndarray | None = None) -> np.ndarray:
        return self._eval(np.zeros(4) if curr_dq is None else curr_dq)[1]


def fit_saturating_pm_flux(cur: np.ndarray, psi: np.ndarray, p0: dict | None = None,
                           forward: bool = True, max_nfev: int = 4000) -> tuple[dict, object]:
    """Least squares in two stages: inverse form (current residuals [A], no Newton)
    for a robust start, then forward form (flux residuals [mWb] through Newton),
    which is the quantity the drive uses. Returns (params, scipy result)."""
    th0 = np.clip(to_theta(p0 if p0 is not None else P0), LO + 1e-9, HI - 1e-9)

    def res_inv(th):
        m = SaturatingPMFlux(to_params(th))
        return (m.current(psi) - cur).ravel()

    r = least_squares(res_inv, th0, bounds=(LO, HI), method="trf", x_scale="jac", max_nfev=max_nfev)
    if not forward:
        return to_params(r.x), r
    warm = {"psi": None}

    def res_fwd(th):
        m = SaturatingPMFlux(to_params(th))
        pred = m.total_flux(cur, warm["psi"])
        warm["psi"] = pred
        return 1e3 * (pred - psi).ravel()

    r = least_squares(res_fwd, r.x, bounds=(LO, HI), method="trf", x_scale="jac", max_nfev=max_nfev)
    return to_params(r.x), r


def pmsm5_prototype_physics(
    curr_max: float | None = None,
    volt_max: float | None = None,
    t_winding: float | None = None,
    campaign: str = _PMSM5_DEFAULT,
) -> PMSMDrive:
    """Five-phase PMSM prototype with the fitted physics-based flux model --- the
    same machine and converter assumptions as ``pmsm5_prototype_fluxmap`` for the
    same ``campaign`` (limits, R_s at the map temperature), but defined beyond the
    measured current grid (field weakening, high torque, generating quadrant by
    mirror symmetry). Use it to complete setpoints the grid does not cover; inside
    the grid it agrees with the measured map to within the fit error recorded in
    the parameter file."""
    _, i_max, v_max, R_s = _pmsm5_setup(campaign, curr_max, volt_max, t_winding)
    p = json.loads((_DATA_DIR / _PMSM5_CAMPAIGNS[campaign]["sat"]).read_text())["params"]
    flux = SaturatingPMFlux(p)
    params = PMSMParams(
        n_phases=5, n_ppairs=2,
        R_stat=R_s * np.eye(4),
        L_stat=flux.inductance(0.0, np.zeros(4)), flux_pm=flux.total_flux(np.zeros((1, 4)))[0],
        curr_max=i_max, volt_max=v_max,
    )
    return PMSMDrive(params, flux=flux)
