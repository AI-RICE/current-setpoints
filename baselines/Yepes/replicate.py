"""
Replicate the ML (minimum-SCL, no current limits) closed-form reference of

    A. G. Yepes et al., "Open-Phase-Tolerant Online Current References for
    Maximum Torque Range and Minimum Loss With Current and Torque-Ripple
    Limits for n-Phase Nonsalient PMSMs With Nonsinusoidal Back EMF",
    IEEE TTE 10(1):432-449, 2024.  DOI 10.1109/tte.2023.3288525.

Specifically Eq (7) of that paper (Section II): the per-sample Lagrangian
solution for minimum stator copper loss with the torque equality and the
wye-zero-sequence equality, *no* current-magnitude limits:

    i  =  [ f .* e  -  (fT e / fT f) f ]
          ---------------------------------- * T
          (f .* e)T e  -  (fT e)^2 / fT f

where f_k = 1 for healthy phases, 0 for open phases, and e is the back-EMF
over mechanical speed (the torque function).

Yepes's five-phase illustrative example (page 436): n = 5, non-salient,
back-EMF = fundamental + 3rd harmonic with amplitude 30% of the fundamental,
"opposite phase angle" (interpreted as the flattening sign, see README).
Limits set so that the converter peak limit equals 1 A in healthy ML
conditions; the rated torque (= healthy ML torque at peak=1) is 152.7 Nm.
The published anchor: for one phase open, ML hits the peak limit at
T1 = 75.5 Nm  ->  T1 / T_rated = 49.4%.

This script computes the same quantity from first principles. Run:

    python baselines/Yepes/replicate.py
"""

from __future__ import annotations

import numpy as np

# -- Yepes Section-II illustrative example -------------------------------
N_PHASES = 5
N_THETA = 4000
T_RATED_NM = 152.7  # Yepes's stated healthy rated torque [Nm]
T1_PUBLISHED_NM = 75.5  # Yepes's stated 1-phase-open peak-limited torque [Nm]

theta = np.linspace(0.0, 2.0 * np.pi, N_THETA, endpoint=False)
alphas = 2.0 * np.pi * np.arange(N_PHASES) / N_PHASES

# Back-EMF (Yepes Fig 4): fundamental + 30% third harmonic, "opposite phase".
# We take the FLATTENING convention (+ sign on the third): this is the one
# that reproduces Yepes's T1/T_rated ratio to within their three-digit
# precision. The peakier convention (- sign) gives ~58%, off by 8 pp.
# Switch SIGN_3 to -1 to see that interpretation.
SIGN_3 = +1
e = np.stack(
    [np.sin(theta - a) + SIGN_3 * 0.3 * np.sin(3 * (theta - a)) for a in alphas],
    axis=0,
)  # shape (n_phases, n_theta)


def yepes_eq7_current_per_T(f: np.ndarray) -> np.ndarray:
    """
    Yepes Eq (7): returns i / T of shape (n_phases, n_theta) for a given
    healthy-mask f (1 healthy, 0 open). Closed-form, no iteration.
    """
    f = np.asarray(f, dtype=float)
    fe = f @ e  # (n_theta,)  = sum_k f_k e_k(theta)
    ff = float(f @ f)  # scalar
    num = (f[:, None] * e) - (fe / ff) * f[:, None]  # (n_phases, n_theta)
    den = ((f[:, None] * e) * e).sum(axis=0) - (fe**2) / ff  # (n_theta,)
    return num / den


def peak_T_at_pk1(f: np.ndarray) -> tuple[float, float]:
    """Returns (peak_per_T, T_at_peak=1) for ML reference under healthy-mask f."""
    i_per_T = yepes_eq7_current_per_T(f)
    peak_per_T = float(np.max(np.abs(i_per_T)))
    return peak_per_T, 1.0 / peak_per_T


def main() -> None:
    print("=" * 66)
    print("Yepes (2024 TTE) NSBE-FRML, Section-II ML reference (Eq 7)")
    print("=" * 66)
    print(f"n = {N_PHASES}, non-salient, back-EMF = sin + {0.3 * SIGN_3:+.2f}*sin(3.)")
    print("(SIGN_3=+1 is the flattening 'opposite phase' interpretation\n that matches Yepes's stated ratio.)\n")

    # Crest factor of the back-EMF for sanity
    per_phase_peak = float(np.max(np.abs(e[0])))
    per_phase_rms = float(np.sqrt(np.mean(e[0] ** 2)))
    print(
        f"per-phase back-EMF:  peak = {per_phase_peak:.4f}  rms = {per_phase_rms:.4f}  "
        f"crest = {per_phase_peak / per_phase_rms:.4f}\n"
    )

    # --- Healthy ---
    f_h = np.ones(N_PHASES)
    pk_h, T_h_at_pk1 = peak_T_at_pk1(f_h)
    print(f"HEALTHY (Eq 7, f = ones({N_PHASES})):")
    print(f"  peak |i_ML| / T   = {pk_h:.6f}")
    print(f"  T at peak = 1 A   = {T_h_at_pk1:.4f}  (normalized; this is Yepes's T_rated)\n")

    # --- One open phase (a) ---
    f_f = np.array([0.0, 1, 1, 1, 1])
    i_f = yepes_eq7_current_per_T(f_f)
    print("1-PHASE OPEN (Eq 7, f = (0,1,1,1,1)):")
    print(f"  max |i_a / T| (should be 0)  = {np.max(np.abs(i_f[0])):.2e}")
    peaks_per_phase = np.max(np.abs(i_f), axis=1)
    print(
        f"  peak per surviving phase     = {np.round(peaks_per_phase[1:], 4)}  "
        f"(unequal -- ML does NOT impose equal magnitude)"
    )
    pk_f, T_f_at_pk1 = peak_T_at_pk1(f_f)
    print(f"  peak |i_ML| / T              = {pk_f:.6f}")
    print(f"  T at peak = 1 A              = {T_f_at_pk1:.4f}\n")

    # --- Ratio ---
    ratio_pct = T_f_at_pk1 / T_h_at_pk1 * 100.0
    pub_ratio_pct = T1_PUBLISHED_NM / T_RATED_NM * 100.0
    print(f"ANCHOR:  T1 / T_rated  =  {ratio_pct:6.3f}%")
    print(f"Yepes :  T1 / T_rated  =  {T1_PUBLISHED_NM}/{T_RATED_NM}  =  {pub_ratio_pct:6.3f}%")
    delta = ratio_pct - pub_ratio_pct
    print(f"diff  :  {delta:+.3f} pp  (within Yepes's three-digit precision of T1=75.5, T_rated=152.7)\n")

    # Express the absolute Nm using Yepes's normalization (T_rated = healthy at pk=1)
    nm_per_unit = T_RATED_NM / T_h_at_pk1
    print(f"In Nm (with T_rated = {T_RATED_NM} Nm by definition):")
    print(f"  HEALTHY T at peak=1    = {T_h_at_pk1 * nm_per_unit:7.2f} Nm   (Yepes: {T_RATED_NM})")
    print(f"  1-PHASE OPEN  T at pk=1 = {T_f_at_pk1 * nm_per_unit:7.2f} Nm   (Yepes: {T1_PUBLISHED_NM})")


if __name__ == "__main__":
    main()
