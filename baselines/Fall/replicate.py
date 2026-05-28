"""
Validate Fall (2016) by implementing Eqs (16)-(19) *verbatim*.

For phase-a open, Fall's surviving currents (Eqs 16-19) are functions of
(i'_d1, i'_q1) constants and rotor angle theta. Let
    N1 = i'_d1 cos(theta) - i'_q1 sin(theta)
    N2 = i'_d1 sin(theta) + i'_q1 cos(theta)
    D1 = cos(2 pi/5) - cos(4 pi/5)
    D2 = sin(2 pi/5) + sin(4 pi/5)
then (Eqs 16-19):
    i'_b = sqrt(5/8) * ( + N1/D1 + N2/D2 )
    i'_c = sqrt(5/8) * ( - N1/D1 + N2/D2 )
    i'_d = sqrt(5/8) * ( - N1/D1 - N2/D2 )
    i'_e = sqrt(5/8) * ( + N1/D1 - N2/D2 )

Analytical sanity checks done by hand:
  - wye: i'_b + i'_c + i'_d + i'_e = 0 (signs cancel).
  - Park back: substituting Eq (14) returns (i'_d1, i'_q1) identically.
  - all 4 surviving peaks share the value
        sqrt(5/8) * sqrt(1/D1^2 + 1/D2^2) * sqrt(i_d1^2 + i_q1^2).

Fall's parameters (Table 1): p=7, phi1=19.4 mWb, Imax=60 A.

NOTE on Eq (11). The paper writes a sqrt-fraction in Eqs (10)/(11)/(12).
Reading the rendered PDF naively as sqrt(5/2) gives a healthy max torque
of only 8.15 Nm at Imax=60 A, contradicting Fall's Fig 3a value of ~20.5
Nm. Back-solving from T = p*sqrt(5/2)*phi1*i_q1 = 20.5 forces i_q1 ~ 95.4
A, which is consistent with peak |i_a| <= 60 A only if the factor in Eq
(11) is sqrt(2/5) ~ 0.632, NOT sqrt(5/2). So the equations use
power-invariant Park sqrt(2/5) in BOTH directions (Eqs 13 forward and
11/12 amplitude-from-dq), the latter as 1/sqrt(5/2) = sqrt(2/5). With
that reading the whole paper is self-consistent and Fig 3a is reproduced.

Fall's torque (Eq 8): T = p*sqrt(5/2)*phi1*i_q1.
Fall's healthy peak (Eq 11, i_d3=i_q3=0): sqrt(2/5)*|i_q1| <= Imax.

This script uses Fall's POWER-INVARIANT convention throughout (the i_q1
values it prints are Fall's numbers, NOT the amplitude-invariant ones used
elsewhere in this repo). Healthy max ~20.5 Nm, 1-phase ~15.4 Nm (75%) are
the Fall anchors we want to hit.

Run:
    python baselines/Fall/replicate.py
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

# -- Fall machine --------------------------------------------------------
P_PAIRS = 7
PHI1 = 19.4e-3  # Wb
IMAX = 60.0  # A
THETA = np.linspace(0.0, 2.0 * np.pi, 4000, endpoint=False)  # dense angle grid

ANG = np.array([0.0, 2 * np.pi / 5, 4 * np.pi / 5, 6 * np.pi / 5, 8 * np.pi / 5])  # phase a..e

D1 = np.cos(2 * np.pi / 5) - np.cos(4 * np.pi / 5)
D2 = np.sin(2 * np.pi / 5) + np.sin(4 * np.pi / 5)
SQRT58 = np.sqrt(5.0 / 8.0)


def torque(i_q1: float) -> float:
    """Fall Eq (8) sinusoidal back-EMF: T = p sqrt(5/2) phi1 i_q1."""
    return P_PAIRS * np.sqrt(2.5) * PHI1 * i_q1


# -- Healthy (Fall Eq 11): peak phase current -----------------------------
def healthy_phase_currents(i_d1: float, i_q1: float, i_d3: float = 0.0, i_q3: float = 0.0):
    """Returns 5xT array of phase currents in Fall's convention (Eq 10/11).

    Eq (11) form: |i_k(theta)| = sqrt(2/5) * |i_d1 cos(theta - alpha_k)
                                              - i_q1 sin(theta - alpha_k)
                                              + i_d3 cos(3(theta-alpha_k))
                                              - i_q3 sin(3(theta-alpha_k))|
    """
    out = np.empty((5, THETA.size))
    for k, a in enumerate(ANG):
        t = THETA - a
        out[k] = np.sqrt(0.4) * (i_d1 * np.cos(t) - i_q1 * np.sin(t) + i_d3 * np.cos(3 * t) - i_q3 * np.sin(3 * t))
    return out


def healthy_Tmax() -> tuple[float, float]:
    """At low speed (no FW): max T s.t. peak phase <= Imax, i_d1 = i_d3 = i_q3 = 0.
    peak = sqrt(2/5) * |i_q1|, so i_q1_max = Imax / sqrt(2/5) = Imax * sqrt(5/2).
    """
    i_q1_max = IMAX * np.sqrt(2.5)
    return torque(i_q1_max), i_q1_max


# -- 1-phase fault (phase a open), Fall Eqs (16)-(19) ---------------------
def fault1_phase_currents(i_d1: float, i_q1: float):
    """Returns 4xT array of surviving phase currents (b, c, d, e) per Eqs 16-19."""
    N1 = i_d1 * np.cos(THETA) - i_q1 * np.sin(THETA)
    N2 = i_d1 * np.sin(THETA) + i_q1 * np.cos(THETA)
    A = N1 / D1
    B = N2 / D2
    i_b = SQRT58 * (+A + B)
    i_c = SQRT58 * (-A + B)
    i_d = SQRT58 * (-A - B)
    i_e = SQRT58 * (+A - B)
    return np.stack([i_b, i_c, i_d, i_e], axis=0)


def fault1_peak(i_d1: float, i_q1: float) -> float:
    """Analytical: sqrt(5/8) * sqrt(1/D1^2 + 1/D2^2) * sqrt(i_d1^2 + i_q1^2)."""
    return SQRT58 * np.sqrt(1.0 / D1**2 + 1.0 / D2**2) * np.sqrt(i_d1**2 + i_q1**2)


def fault1_Tmax_low_speed() -> float:
    """At low speed: max T s.t. peak phase <= Imax, i_d1 = 0 (no FW)."""
    # peak = K * |i_q1| with K = sqrt(5/8) * sqrt(1/D1^2 + 1/D2^2)
    K = SQRT58 * np.sqrt(1.0 / D1**2 + 1.0 / D2**2)
    i_q1_max = IMAX / K
    return torque(i_q1_max)


# -- 1-phase fault: numerical optimization (verify analytical) ------------
def fault1_Tmax_numerical() -> float:
    """SLSQP: maximize T = sqrt(5/2)*p*phi1*i_q1 s.t. max|i'_k(theta)| <= Imax."""

    def neg_T(x):
        return -P_PAIRS * np.sqrt(2.5) * PHI1 * x[1]  # x = (i_d1, i_q1)

    def neg_T_jac(x):
        return np.array([0.0, -P_PAIRS * np.sqrt(2.5) * PHI1])

    # peak constraint via sample grid
    def peak_constraints(x):
        i_d1, i_q1 = x
        I_ph = fault1_phase_currents(i_d1, i_q1)  # 4 x T
        return IMAX - np.max(np.abs(I_ph))  # >= 0

    cons = [{"type": "ineq", "fun": peak_constraints}]
    x0 = np.array([0.0, 30.0])
    res = minimize(neg_T, x0, jac=neg_T_jac, method="SLSQP", constraints=cons, options={"ftol": 1e-10, "maxiter": 1000})
    return -res.fun


# -- 2-phase fault: uniquely-determined surviving currents from -----------
# Park (2 rows of Eq 14/26/35) + wye (1 row) on 3 surviving phases.
# Fall drops the equal-magnitude criterion in this case (Sec 3.3).
def fault2_phase_currents(i_d1: float, i_q1: float, alphas: tuple[float, float, float]):
    """Returns 3xT array of surviving phase currents (in the order of alphas).

    Solves [i_d1, i_q1, 0] = M(theta) @ [i_k1, i_k2, i_k3] for each theta,
    where M's rows are Park-d1, Park-q1, wye on the 3 surviving phases.
    """
    a1, a2, a3 = alphas
    I_ph = np.empty((3, THETA.size))
    sq25 = np.sqrt(0.4)
    for ti, th in enumerate(THETA):
        M = np.array(
            [
                [sq25 * np.cos(th - a1), sq25 * np.cos(th - a2), sq25 * np.cos(th - a3)],
                [-sq25 * np.sin(th - a1), -sq25 * np.sin(th - a2), -sq25 * np.sin(th - a3)],
                [1.0, 1.0, 1.0],
            ]
        )
        I_ph[:, ti] = np.linalg.solve(M, np.array([i_d1, i_q1, 0.0]))
    return I_ph


def fault2_peak(alphas, i_q1_test: float = 50.0) -> float:
    """Empirical: peak surviving phase current per |i_q1| at i_d1 = 0."""
    I_ph = fault2_phase_currents(0.0, i_q1_test, alphas)
    return float(np.max(np.abs(I_ph)) / abs(i_q1_test))


def fault2_Tmax(alphas) -> float:
    K = fault2_peak(alphas)  # peak per |i_q1|
    i_q1_max = IMAX / K
    return torque(i_q1_max)


def main() -> None:
    print("==============================================")
    print("Fall (2016) -- verbatim Eqs (16)-(19) replication")
    print("==============================================")
    print(f"Machine: p={P_PAIRS}, phi1={PHI1 * 1000:.1f} mWb, Imax={IMAX} A")
    print(f"D1 = cos(2pi/5) - cos(4pi/5) = {D1:.6f}")
    print(f"D2 = sin(2pi/5) + sin(4pi/5) = {D2:.6f}\n")

    # Healthy ------------------------------------------------------
    T_h, i_q1_h = healthy_Tmax()
    print(f"HEALTHY:                       T_max = {T_h:7.3f} Nm   i_q1 = {i_q1_h:6.2f} A   (Fall Fig 3a: 20.5 Nm)")

    # 1-phase fault (analytical) ----------------------------------
    K1 = SQRT58 * np.sqrt(1.0 / D1**2 + 1.0 / D2**2)
    T_1ph_anal = fault1_Tmax_low_speed()
    print(
        f"1-PHASE OPEN  (analytic):      T_max = {T_1ph_anal:7.3f} Nm   "
        f"peak/|i_q1| = {K1:.4f}   ratio = {T_1ph_anal / T_h * 100:.2f}%   "
        f"(Fall: 15.4 Nm, ~75%)"
    )

    # 1-phase fault (numerical: free (i_d1, i_q1)) ----------------
    T_1ph_num = fault1_Tmax_numerical()
    print(f"1-PHASE OPEN  (numerical):     T_max = {T_1ph_num:7.3f} Nm   ratio = {T_1ph_num / T_h * 100:.2f}%")

    # Verify all 4 surviving peaks are equal (Eqs 16-19 property)
    i_q1 = 50.0
    I_ph = fault1_phase_currents(0.0, i_q1)
    peaks = np.max(np.abs(I_ph), axis=1)
    rmss = np.sqrt(np.mean(I_ph**2, axis=1))
    print(f"\nVerify equal magnitude at (i_d1, i_q1) = (0, {i_q1}):")
    print(f"  peaks per phase (b,c,d,e): {np.round(peaks, 3)}   (should all be equal)")
    print(f"  rms   per phase (b,c,d,e): {np.round(rmss, 3)}    (should all be equal)")
    print(f"  expected peak = K * |i_q1| = {K1 * i_q1:.3f}")

    # 2-phase adjacent (phases a, b open; surviving c, d, e at 4pi/5, 6pi/5, 8pi/5)
    a_adj = (4 * np.pi / 5, 6 * np.pi / 5, 8 * np.pi / 5)
    T_2a = fault2_Tmax(a_adj)
    I_adj = fault2_phase_currents(0.0, 50.0, a_adj)
    print(
        f"\n2-PHASE ADJACENT  (a, b open):  T_max = {T_2a:7.3f} Nm   "
        f"ratio = {T_2a / T_h * 100:.2f}%   (Fall: ~5.5 Nm, ~27%)"
    )
    print(f"  peaks per surv phase (c,d,e):  {np.round(np.max(np.abs(I_adj), axis=1), 3)}")

    # 2-phase non-adjacent (phases a, c open; surviving b, d, e at 2pi/5, 6pi/5, 8pi/5)
    a_non = (2 * np.pi / 5, 6 * np.pi / 5, 8 * np.pi / 5)
    T_2n = fault2_Tmax(a_non)
    I_non = fault2_phase_currents(0.0, 50.0, a_non)
    print(
        f"\n2-PHASE NON-ADJ   (a, c open):  T_max = {T_2n:7.3f} Nm   "
        f"ratio = {T_2n / T_h * 100:.2f}%   (Fall: ~9 Nm, ~44%)"
    )
    print(f"  peaks per surv phase (b,d,e):  {np.round(np.max(np.abs(I_non), axis=1), 3)}")


if __name__ == "__main__":
    main()
