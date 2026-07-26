from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .machines import DriveModel, IM9Phase, _build_cross_coupling


@dataclass
class IMHarmonicParams:
    """Electrical parameters of one odd-harmonic plane (h = 1, 3, ...)."""

    R_r: float
    L_mu: float  # rated (unsaturated) magnetizing inductance [H]
    L_s_sigma: float
    L_r_sigma: float
    # optional magnetizing-saturation lookup table: L_mu as a function of the
    # plane's magnetizing current; constant L_mu when absent
    i_mag_table: np.ndarray | None = None  # strictly increasing [A]
    L_mu_table: np.ndarray | None = None  # matching L_mu values [H]

    def __post_init__(self) -> None:
        if (self.i_mag_table is None) != (self.L_mu_table is None):
            raise ValueError("i_mag_table and L_mu_table must be given together.")
        if self.i_mag_table is not None:
            self.i_mag_table = np.asarray(self.i_mag_table, dtype=float)
            self.L_mu_table = np.asarray(self.L_mu_table, dtype=float)
            if self.i_mag_table.ndim != 1 or self.i_mag_table.shape != self.L_mu_table.shape:
                raise ValueError("Saturation tables must be 1-D and of equal length.")
            if np.any(np.diff(self.i_mag_table) <= 0):
                raise ValueError("i_mag_table must be strictly increasing.")

    def L_mu_at(self, i_mag: float) -> float:
        if self.i_mag_table is None:
            return self.L_mu
        return float(np.interp(abs(i_mag), self.i_mag_table, self.L_mu_table))


@dataclass
class IMLUTParams:
    """Complete machine description: identity, electrical data per harmonic
    plane, and the converter's current/voltage limits. One instance == one
    physical machine + converter. Deliberately NOT here: the speed horizon
    of a setpoint map (opts["omega_max"] at grid time) — that is a property
    of the optimization run, not of the drive."""

    n_phases: int
    n_ppairs: int
    R_s: float
    harmonics: list[IMHarmonicParams]  # index i <-> harmonic h = 2 i + 1
    curr_max: float
    volt_max: float
    k_v: float = 0.0
    k_h: float = 0.0
    # fixed rotor constants for the slip relation omega_r =
    # (R_r/L_r)(i_sd1/i_sq1); when None the plane-1 values (with the
    # saturation-dependent L_mu) are used. FEM characterizations typically
    # prescribe slip with FIXED no-load constants — set these to reproduce
    # that convention exactly.
    slip_R_r: float | None = None
    slip_L_r: float | None = None


class IMDriveLUT(DriveModel):
    """Induction drive whose machine data arrive entirely through `params` —
    the class holds physics only; a concrete machine is an instance.

    Saturation: each plane's L_mu is read from its lookup table at the
    plane's d-axis (magnetizing) current under rotor-flux orientation;
    without a table the plane is linear and reproduces `IM9Phase` exactly.
    """

    def __init__(self, params: IMLUTParams) -> None:
        self.params = params
        self.n_phases = params.n_phases
        self.n_harmonics = len(params.harmonics)
        self.dim = 2 * self.n_harmonics
        self.n_ppairs = params.n_ppairs
        self.k_phase = params.n_phases / 2

        self.R_stat = params.R_s * np.eye(self.dim)
        self.k_v = params.k_v
        self.k_h = params.k_h
        self.curr_max = params.curr_max
        self.volt_max = params.volt_max

        self._cross_coupling = _build_cross_coupling(self.n_harmonics)

    def _plane_L_mu(self, i: int, curr_dq: np.ndarray | None) -> float:
        i_mag = 0.0 if curr_dq is None else float(curr_dq[2 * i])
        return self.params.harmonics[i].L_mu_at(i_mag)

    def _L_matrices(self, curr_dq: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        L_mu = np.zeros((self.dim, self.dim))
        L_s = np.zeros((self.dim, self.dim))
        L_r = np.zeros((self.dim, self.dim))
        R_r = np.zeros((self.dim, self.dim))
        eye2 = np.eye(2)
        for i, p in enumerate(self.params.harmonics):
            L_mu_h = self._plane_L_mu(i, curr_dq)
            sl = slice(2 * i, 2 * i + 2)
            L_mu[sl, sl] = L_mu_h * eye2
            L_s[sl, sl] = (L_mu_h + p.L_s_sigma) * eye2
            L_r[sl, sl] = (L_mu_h + p.L_r_sigma) * eye2
            R_r[sl, sl] = p.R_r * eye2
        return L_mu, L_s, L_r, R_r

    def k_ir(self, omega_r: float, curr_dq: np.ndarray | None = None) -> np.ndarray:
        K = np.zeros((self.dim, self.dim))
        for i, p in enumerate(self.params.harmonics):
            h = 2 * i + 1
            L_mu_h = self._plane_L_mu(i, curr_dq)
            K[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = IM9Phase._k_ir_block(
                omega_r, L_mu_h, L_mu_h + p.L_r_sigma, p.R_r, h
            )
        return K

    def slip_from_dq_foc(self, curr_dq: np.ndarray, eps: float = 1e-9) -> float:
        i_sd_1 = curr_dq[0]
        i_sq_1 = curr_dq[1]
        if abs(i_sq_1) < eps:
            return 0.0
        p1 = self.params.harmonics[0]
        R_r_1 = self.params.slip_R_r if self.params.slip_R_r is not None else p1.R_r
        if self.params.slip_L_r is not None:
            L_r_1 = self.params.slip_L_r
        else:
            L_r_1 = self._plane_L_mu(0, curr_dq) + p1.L_r_sigma
        return (R_r_1 / L_r_1) * (i_sd_1 / i_sq_1)

    def voltage_operator(self, omega: float, curr_dq: np.ndarray) -> np.ndarray:
        return self.R_stat + omega * self._cross_coupling @ self.inductance(omega, curr_dq)

    def bemf_dq(self, omega: float, curr_dq: np.ndarray) -> np.ndarray:
        return np.zeros(self.dim)

    def inductance(self, omega: float, curr_dq: np.ndarray | None = None) -> np.ndarray:
        L_mu, L_s, _, _ = self._L_matrices(curr_dq)
        if curr_dq is None:
            return L_s
        omega_r = self.slip_from_dq_foc(curr_dq)
        return L_s + L_mu @ self.k_ir(omega_r, curr_dq)

    def torque(self, omega: float, curr_dq: np.ndarray) -> float:
        omega_r = self.slip_from_dq_foc(curr_dq)
        L_mu, _, _, _ = self._L_matrices(curr_dq)
        A = (self.n_phases * self.n_ppairs / 2.0) * (self._cross_coupling @ L_mu @ self.k_ir(omega_r, curr_dq))
        return float(curr_dq @ A @ curr_dq)

    def seeds(self, guess: np.ndarray | None = None) -> list[np.ndarray]:
        dim = self.dim
        i_max = self.curr_max
        candidates: list[np.ndarray] = []
        if guess is not None:
            candidates.append(guess.copy())
        c = np.zeros(dim)
        c[0] = 0.1
        c[1] = 0.1
        candidates.append(c)
        for amp in (0.3, 0.5, 0.7):
            c = np.zeros(dim)
            c[0] = i_max * amp
            c[1] = i_max * amp
            candidates.append(c)
        c = np.zeros(dim)
        c[0] = i_max * 0.85
        c[1] = i_max * 0.30
        candidates.append(c)
        if dim >= 4:
            c = np.zeros(dim)
            c[0] = i_max * 0.7
            c[1] = i_max * 0.7
            c[2] = i_max * 0.1
            c[3] = -i_max * 0.1
            candidates.append(c)
        return candidates

    def copper_loss(self, omega: float, curr_dq: np.ndarray) -> float:
        P_stator = self.k_phase * self.params.R_s * float(curr_dq @ curr_dq)
        omega_r = self.slip_from_dq_foc(curr_dq)
        _, _, _, R_r = self._L_matrices(curr_dq)
        i_r = self.k_ir(omega_r, curr_dq) @ curr_dq
        P_rotor = self.k_phase * float(i_r @ R_r @ i_r)
        return P_stator + P_rotor


def im5_async(curr_max: float = 13.5, volt_max: float = 325.0) -> IMDriveLUT:
    """The five-phase IM of the 2025-5f-async paper (IM_5f.tex,
    tab:machine_parameters), linear magnetics: m=5, p_p=2, R_s=0.74;
    per-harmonic (h=1, h=3): R_r=0.61/0.48, L_mu=367/36.1 mH,
    L_s_sigma=7.23/8.94 mH, L_r_sigma=5.07/3.22 mH. Default limits are the
    paper's I_max=13.5 A, V_max=325 V. The setpoint-map speed horizon is
    supplied at grid time (opts["omega_max"]), not here."""
    params = IMLUTParams(
        n_phases=5,
        n_ppairs=2,
        R_s=0.74,
        harmonics=[
            IMHarmonicParams(R_r=0.61, L_mu=367.0e-3, L_s_sigma=7.23e-3, L_r_sigma=5.07e-3),
            IMHarmonicParams(R_r=0.48, L_mu=36.1e-3, L_s_sigma=8.94e-3, L_r_sigma=3.22e-3),
        ],
        curr_max=curr_max,
        volt_max=volt_max,
    )
    return IMDriveLUT(params)


# Five-phase, Tesla1-class induction machine (150 kW eMobility design; the
# 5-phase variant of the motor used in the first-generation Tesla drive).
#
# Equivalent-circuit parameters COMPUTED (not estimated) from the Ansys FEM
# model by J. Laksar, e-mail "Parametry 5fazoveho asynchronniho stroje z
# Ansysu" (2026-07-20), workbook e5_eMobility_IM_150kW_mod_5ph_h3_linear.xlsx.
# Source uses the Czech machine convention: subscript 1 = stator, 2' = rotor;
# L_1 = L_mu + L_1sigma (total stator), L_2' = L_mu + L_2sigma' (total rotor).
# m = 5, p_p = 3 — so the h=3 plane sees 3*p_p = 9 pole pairs, which is the
# "p = 9" printed on the workbook's `momentovka` (torque-slip) sheet.
#
# These are single-operating-point (lightly saturated) LINEAR parameters. The
# leakages drift with saturation, but for h=1 they are small so the error is
# minor (Laksar). Validated against the 357-point Ansys torque sweep
# I_combs_Results_correct_wr_definition.xlsx, clean slice Id1 = 40 A, no 3rd
# harmonic: T_model matches T_FEM to 0-2 % at rated load (Iq1 >= 160 A). At low
# load the FEM torque runs ~20 % below the linear model — the FEM torque/Iq1
# ratio *rises* with current (a cross-saturation feature a linear model cannot
# capture), consistent with Laksar's own observation on the sweep.
_TESLA5F_R_S = 0.02194121  # Ohm, stator (shared by both planes)
# fundamental plane (h = 1)
_TESLA5F_H1_R_R = 0.006191796  # Ohm, rotor
_TESLA5F_H1_L_MU = 2.193562e-3  # H, magnetizing
_TESLA5F_H1_L_SS = 5.913e-5  # H, stator leakage
_TESLA5F_H1_L_RS = 6.851e-5  # H, rotor leakage
# third-harmonic plane (h = 3)
_TESLA5F_H3_R_R = 0.004503021  # Ohm
_TESLA5F_H3_L_MU = 0.202479e-3  # H
_TESLA5F_H3_L_SS = 6.465e-5  # H
_TESLA5F_H3_L_RS = 5.166e-5  # H

# Fundamental-plane magnetizing saturation, keyed on the d-axis (magnetizing)
# current Id1. Fröhlich-type law L_mu1(Id1) = L0 / (1 + (Id1/i_sat)^p), the
# 3-parameter "Route B" form identified by least-squares against the full
# 357-point Ansys torque sweep (I_combs_Results_correct_wr_definition.xlsx).
# It cuts clean-slice torque RMS from 154 Nm (linear) to 10.8 Nm; L0 recovers
# the computed unsaturated EC at low current (L_mu1(40 A) = 2.17 mH ~ 2.19 mH).
# Chosen over the material B(H) curve and the FEM-measured lambda_d1/Id1 table
# (both ~18 Nm RMS) because those anchor to the *unloaded* magnetizing curve;
# the residual is genuine 2-D cross-saturation. See ADR 0003.
_TESLA5F_H1_SAT_L0 = 2.475684e-3  # H, unsaturated magnetizing inductance
_TESLA5F_H1_SAT_ISAT = 94.7304  # A, saturation knee
_TESLA5F_H1_SAT_P = 2.27219  # -, saturation sharpness
# d-axis current knots at which the law is sampled into the plane's lookup
# table (np.interp clamps flat beyond the top knot).
_TESLA5F_SAT_I_MAG = np.arange(0.0, 300.0 + 1.0, 10.0)  # A


def _tesla5f_h1_L_mu_table() -> np.ndarray:
    i = _TESLA5F_SAT_I_MAG
    return _TESLA5F_H1_SAT_L0 / (1.0 + (i / _TESLA5F_H1_SAT_ISAT) ** _TESLA5F_H1_SAT_P)


def im5_tesla_gen1(
    curr_max: float = 200.0, volt_max: float = 230.0, saturating: bool = True
) -> IMDriveLUT:
    """Five-phase, Tesla1-class 150 kW induction machine — equivalent circuit
    computed from the Ansys FEM model (Laksar, 2026-07-20).

    m = 5, p_p = 3, R_s = 21.94 mOhm. Per harmonic plane (h=1 / h=3):
    R_r = 6.19 / 4.50 mOhm, L_mu = 2.194 / 0.2025 mH, L_s_sigma = 59.1 / 64.7
    uH, L_r_sigma = 68.5 / 51.7 uH. Converter limits default to the FEM
    sweep's I_max = 200 A (a sweep bound, not a thermal rating) and
    phase-voltage amplitude V_max = 230 V.

    saturating (default True): give the fundamental plane the identified
    magnetizing-saturation law L_mu1(Id1) (Route B; see ADR 0003), which
    tracks the 357-point Ansys torque sweep to ~11 Nm RMS. This machine's
    iron is materially saturated over the sweep, so the linear form overshoots
    torque by up to ~3x at high Id1 (RMS 154 Nm) — it is kept (saturating=False)
    only for the small-signal / low-Id1 regime. The h=3 plane stays linear in
    both modes (its saturation is not yet identified).

    The slip relation uses the fundamental no-load constants R_r1 and
    L_r1 = L_mu1 + L_r_sigma1 = 2.262 mH, matching the FEM's omega_r
    definition (held fixed, unaffected by the saturation state). Unlike the
    earlier assumed set, the h=3 plane's small L_r_sigma3 (< L_mu3) gives a
    strong rotor coupling, so this machine supports meaningful 3rd-harmonic
    torque."""
    L_r1 = _TESLA5F_H1_L_MU + _TESLA5F_H1_L_RS  # no-load rotor inductance
    h1_sat = (
        dict(i_mag_table=_TESLA5F_SAT_I_MAG, L_mu_table=_tesla5f_h1_L_mu_table())
        if saturating
        else {}
    )
    params = IMLUTParams(
        n_phases=5,
        n_ppairs=3,
        R_s=_TESLA5F_R_S,
        harmonics=[
            IMHarmonicParams(
                R_r=_TESLA5F_H1_R_R,
                L_mu=_TESLA5F_H1_SAT_L0 if saturating else _TESLA5F_H1_L_MU,
                L_s_sigma=_TESLA5F_H1_L_SS,
                L_r_sigma=_TESLA5F_H1_L_RS,
                **h1_sat,
            ),
            IMHarmonicParams(
                R_r=_TESLA5F_H3_R_R,
                L_mu=_TESLA5F_H3_L_MU,
                L_s_sigma=_TESLA5F_H3_L_SS,
                L_r_sigma=_TESLA5F_H3_L_RS,
            ),
        ],
        curr_max=curr_max,
        volt_max=volt_max,
        slip_R_r=_TESLA5F_H1_R_R,
        slip_L_r=L_r1,
    )
    return IMDriveLUT(params)


def im9_prototype(curr_max: float = 20.0, volt_max: float = 200.0) -> IMDriveLUT:
    """The 9-phase induction prototype, as an instance: identical numbers to
    `IM9Phase.__init__` (identified on the lab machine); converter limits
    default to the values used across the test suite."""
    params = IMLUTParams(
        n_phases=9,
        n_ppairs=2,
        R_s=5.0,
        harmonics=[
            IMHarmonicParams(R_r=1.54, L_mu=496.0e-3, L_s_sigma=15.1e-3, L_r_sigma=53.3e-3),
            IMHarmonicParams(R_r=1.57, L_mu=58.2e-3, L_s_sigma=13.6e-3, L_r_sigma=33.4e-3),
        ],
        curr_max=curr_max,
        volt_max=volt_max,
    )
    return IMDriveLUT(params)
