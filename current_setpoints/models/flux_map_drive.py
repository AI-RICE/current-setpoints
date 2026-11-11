"""Machine-agnostic direct flux-map drive.

Given any sampled current->flux-linkage map psi(i), this model gives torque and
terminal voltage straight from the co-energy relations, independent of an
equivalent circuit:

    T = (m*p_p/2) * i . J . psi        (J = cross-coupling, embeds harmonic h)
    v = R_s*i + omega * J . psi

It works for IM or PMSM alike (it only needs a flux map). For the Tesla1-class
5f IM it reproduces the 357-point Ansys torque sweep to ~0.6 Nm RMS.

Scope: valid inside the convex hull of the sampled currents. `hull_margins`
exposes the facet inequalities so `HullConstrainedOptimizer` can keep setpoints
inside the FEM envelope (no extrapolation).

The machine type enters only through how the samples are used, not the physics:

- IM (defaults): odd symmetry psi(-i) = -psi(i) (`_canon`), so only the
  positive-Id1 half of the data is needed, and the known point psi(0) = 0 is
  added (`anchor_origin`).
- PMSM: ``symmetry="none"`` and ``anchor_origin=False`` -- psi(0) is the PM
  flux, and the data usually covers only the motoring quadrant (i_d1 <= 0).
  ``seed_mode="data"`` takes the optimizer seeds from the sampled points.

Both IM-only choices refuse PMSM-like data loudly instead of interpolating
garbage.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, RBFInterpolator
from scipy.spatial import ConvexHull

from .forward_model import InverterDrop
from .iron_loss import CoreLossLUT, IronLossModel, SteinmetzIronLoss
from .machines import DriveModel, _build_cross_coupling

_DATA = Path(__file__).parent / "data" / "tesla5f_fluxmap.npz"
_DATA_V2 = Path(__file__).parent / "data" / "tesla5f_fluxmap_v2.npz"  # 2026-09 full sweep, |I1|<=470 A, + rotor loss

# Tesla1-class 5f iron-loss coefficients (P = k1|psi1|^2 + k3|psi3|^2), fit to
# the FEM CoreLoss column (R^2=0.994, LOO 2.5%). See ADR 0003.
_TESLA5F_IRON_K1 = 4.75427e4  # W / Wb^2
_TESLA5F_IRON_K3 = 1.353636e6  # W / Wb^2


class FluxMapDrive(DriveModel):
    """Drive defined by a sampled flux-linkage map (Delaunay-linear backend)."""

    def __init__(
        self,
        currents: np.ndarray,
        flux: np.ndarray,
        n_phases: int,
        n_ppairs: int,
        R_s: float,
        curr_max: float,
        volt_max: float,
        iron_loss_model: IronLossModel | None = None,
        flux_backend: str = "smooth",
        rotor_loss_model: IronLossModel | None = None,
        symmetry: str = "odd",
        anchor_origin: bool = True,
        seed_mode: str = "fixed",
        flux_smoothing: float = 0.0,
    ) -> None:
        currents = np.asarray(currents, float)
        flux = np.asarray(flux, float)
        if symmetry not in ("odd", "none"):
            raise ValueError(f"symmetry must be 'odd' or 'none', got {symmetry!r}")
        if symmetry == "odd" and np.any(currents[:, 0] < 0):
            raise ValueError(
                "symmetry='odd' folds every query onto i_d1 >= 0, but the map has samples at "
                "i_d1 < 0 (e.g. a PMSM map) -- pass symmetry='none'"
            )
        i_mag, psi_mag = np.linalg.norm(currents, axis=1), np.linalg.norm(flux, axis=1)
        k0 = int(np.argmin(i_mag))
        if anchor_origin and i_mag[k0] < 0.05 * i_mag.max() and psi_mag[k0] > 0.2 * psi_mag.max():
            raise ValueError(
                "the map has a sample at (near) i = 0 carrying large flux (PM flux?), "
                "contradicting the psi(0) = 0 anchor -- pass anchor_origin=False"
            )
        if seed_mode not in ("fixed", "data"):
            raise ValueError(f"seed_mode must be 'fixed' or 'data', got {seed_mode!r}")
        self.symmetry = symmetry
        self.seed_mode = seed_mode
        self.rotor_loss_model = rotor_loss_model  # FEM rotor-cage Joule loss LUT (W), if the map has it
        self.n_phases = n_phases
        self.n_harmonics = 2
        self.dim = 4
        self.n_ppairs = n_ppairs
        self.k_phase = n_phases / 2
        self.R_stat = R_s * np.eye(self.dim)
        self.k_v = 0.0
        self.k_h = 0.0
        self.curr_max = curr_max
        self.volt_max = volt_max
        self.iron_loss_model = iron_loss_model
        self.flux_backend = flux_backend
        self._J = _build_cross_coupling(self.n_harmonics)
        if anchor_origin:
            # origin-anchored so low-current queries stay inside the hull
            pts = np.vstack([np.zeros((1, 4)), currents])
            val = np.vstack([np.zeros((1, 4)), flux])
        else:
            pts, val = currents, flux
        self._hull_eq = ConvexHull(pts).equations
        if flux_backend == "smooth":
            # C-infinity thin-plate-spline: unbiased at the curved peak-torque
            # edge where Delaunay-linear reads ~5% low (see ADR 0003).
            # flux_smoothing > 0 trades exact interpolation for noise rejection
            # (measured maps).
            self._rbf = RBFInterpolator(pts, val, kernel="thin_plate_spline", smoothing=flux_smoothing)
        elif flux_backend == "linear":
            self._lin = LinearNDInterpolator(pts, val)
            self._near = NearestNDInterpolator(pts, val)
        else:
            raise ValueError(f"flux_backend must be 'smooth' or 'linear', got {flux_backend!r}")
        # a Delaunay membership test for `in_hull`, regardless of backend
        self._member = LinearNDInterpolator(pts, val[:, :1])
        # sampled torques, for data-driven seeds
        self._samples = pts
        self._sample_torque = self.n_phases * self.n_ppairs / 2.0 * np.einsum("ni,ij,nj->n", pts, self._J, val)

    @classmethod
    def from_npz(
        cls,
        path,
        *,
        n_phases: int,
        n_ppairs: int,
        R_s: float,
        curr_max: float,
        volt_max: float,
        iron_loss: str | None = None,
        iron_k: tuple[float, float] | None = None,
        flux_backend: str = "smooth",
        symmetry: str = "odd",
        anchor_origin: bool = True,
        seed_mode: str = "fixed",
        flux_smoothing: float = 0.0,
    ) -> "FluxMapDrive":
        """Build from an ``.npz`` flux map --- machine-agnostic, point it at any
        machine's data. Required arrays: ``currents`` [N,4] and ``flux`` [N,4]
        (dq of harmonics 1,3); ``P_core`` [N] is needed only for
        ``iron_loss="lut"``. ``iron_loss``: ``None``, ``"lut"`` (uses
        ``P_core``), or ``"steinmetz"`` (needs ``iron_k=(k1, k3)``).
        ``symmetry``/``anchor_origin``/``seed_mode``/``flux_smoothing``: see
        the module docstring (defaults = IM map)."""
        z = np.load(path)
        currents, flux = z["currents"], z["flux"]
        rlm = CoreLossLUT(currents, z["P_rotor"]) if "P_rotor" in z.files else None
        if iron_loss == "steinmetz":
            if iron_k is None:
                raise ValueError("iron_loss='steinmetz' requires iron_k=(k1, k3)")
            ilm: IronLossModel | None = SteinmetzIronLoss(*iron_k)
        elif iron_loss == "lut":
            ilm = CoreLossLUT(currents, z["P_core"])
        elif iron_loss is None:
            ilm = None
        else:
            raise ValueError(f"iron_loss must be 'steinmetz', 'lut' or None, got {iron_loss!r}")
        return cls(
            currents=currents, flux=flux, n_phases=n_phases, n_ppairs=n_ppairs,
            R_s=R_s, curr_max=curr_max, volt_max=volt_max,
            iron_loss_model=ilm, flux_backend=flux_backend, rotor_loss_model=rlm,
            symmetry=symmetry, anchor_origin=anchor_origin, seed_mode=seed_mode,
            flux_smoothing=flux_smoothing,
        )

    def _canon(self, curr_dq: np.ndarray) -> tuple[float, np.ndarray]:
        c = np.asarray(curr_dq, float)
        if self.symmetry == "odd" and c[0] < 0:
            return -1.0, -c
        return 1.0, c

    def flux(self, omega: float, curr_dq: np.ndarray) -> np.ndarray:
        s, cc = self._canon(curr_dq)
        q = cc.reshape(1, 4)
        if self.flux_backend == "smooth":
            return s * self._rbf(q)[0]
        psi = self._lin(q)[0]
        if np.any(np.isnan(psi)):
            psi = self._near(q)[0]
        return s * psi

    def in_hull(self, curr_dq: np.ndarray) -> bool:
        _, cc = self._canon(curr_dq)
        return not np.any(np.isnan(self._member(cc.reshape(1, 4))[0]))

    def hull_margins(self, curr_dq: np.ndarray) -> np.ndarray:
        """Signed facet distances of the sampled-current hull; all >= 0 iff
        inside. Used by HullConstrainedOptimizer to forbid extrapolation."""
        x = np.asarray(curr_dq, float)
        return -(self._hull_eq[:, :-1] @ x + self._hull_eq[:, -1])

    def torque(self, omega: float, curr_dq: np.ndarray) -> float:
        psi = self.flux(omega, curr_dq)
        return float(self.n_phases * self.n_ppairs / 2.0 * curr_dq @ self._J @ psi)

    def bemf_dq(self, omega: float, curr_dq: np.ndarray) -> np.ndarray:
        return omega * self._J @ self.flux(omega, curr_dq)

    def voltage_operator(self, omega: float, curr_dq: np.ndarray) -> np.ndarray:
        # v = R_stat @ i + bemf_dq; the flux-induced part lives in bemf_dq
        return self.R_stat

    def inductance(self, omega: float, curr_dq: np.ndarray | None = None) -> np.ndarray:
        """Differential (tangent) inductance ``L_d = dpsi/di`` at the operating
        point, via a central finite difference of the flux map (symmetrised for
        Maxwell reciprocity). This is the proper small-signal inductance --- not
        a chord/secant. With :meth:`flux_offset` it splits ``psi = psi* + L_d@i``
        into a PM-like offset and a linear part (cf. App. C, 5f-async)."""
        x = np.zeros(self.dim) if curr_dq is None else np.asarray(curr_dq, float)
        h = 1e-3 * max(self.curr_max, 1.0)
        jac = np.empty((self.dim, self.dim))
        for b in range(self.dim):
            e = np.zeros(self.dim)
            e[b] = h
            jac[:, b] = (self.flux(omega, x + e) - self.flux(omega, x - e)) / (2.0 * h)
        return 0.5 * (jac + jac.T)

    def flux_offset(self, omega: float, curr_dq: np.ndarray) -> np.ndarray:
        """PM-like flux offset ``psi* = psi(i) - L_d@i`` (the flux not explained
        by the local differential inductance); the back-EMF gain is
        ``omega * J @ psi*``. By construction ``psi(i) == flux_offset + L_d@i``."""
        x = np.asarray(curr_dq, float)
        return self.flux(omega, x) - self.inductance(omega, x) @ x

    def copper_loss(self, omega: float, curr_dq: np.ndarray) -> float:
        """Stator copper loss, plus the FEM rotor-cage Joule loss when the map
        carries a ``P_rotor`` column (v2 map); stator-only otherwise."""
        p = self.k_phase * self.R_stat[0, 0] * float(curr_dq @ curr_dq)
        if self.rotor_loss_model is not None:
            p += self.rotor_loss_model.loss(omega, curr_dq, None)
        return p

    def iron_loss_at(self, omega: float, curr_dq: np.ndarray) -> float:
        if self.iron_loss_model is None:
            raise ValueError("no iron_loss_model set on this drive")
        return self.iron_loss_model.loss(omega, curr_dq, self.flux(omega, curr_dq))

    def seeds(self, guess: np.ndarray | None = None) -> list[np.ndarray]:
        i_max = self.curr_max
        c: list[np.ndarray] = []
        if guess is not None:
            c.append(np.asarray(guess, float).copy())
        if self.seed_mode == "data":
            # highest-torque sample within each current-magnitude band: always
            # inside the sampled region, whatever quadrant the machine uses
            mag = np.linalg.norm(self._samples, axis=1)
            for frac in (0.25, 0.5, 0.75, 1.0):
                band = mag <= frac * mag.max() + 1e-9
                c.append(self._samples[band][np.argmax(self._sample_torque[band])].copy())
            return c
        c.append(np.array([0.1, 0.1, 0.0, 0.0]))
        for amp in (0.3, 0.5, 0.7):
            c.append(np.array([i_max * amp, i_max * amp, 0.0, 0.0]))
        c.append(np.array([i_max * 0.85, i_max * 0.30, 0.0, 0.0]))
        c.append(np.array([i_max * 0.7, i_max * 0.7, i_max * 0.1, -i_max * 0.1]))
        return c


def im5_tesla_gen1_fluxmap(
    curr_max: float = 200.0,
    volt_max: float = 230.0,
    iron_loss: str | None = "steinmetz",
    flux_backend: str = "smooth",
    data_path=_DATA,
) -> FluxMapDrive:
    """Tesla1-class five-phase IM as a direct FEM flux-map drive --- the bundled
    EXAMPLE of :class:`FluxMapDrive` (and a usage template for your own machine).

    Default data: 357-point Ansys sweep (`data/tesla5f_fluxmap.npz`: currents,
    flux, T_fem, P_core). m=5, p_p=3, R_s=21.94 mOhm (computed EC; ADR 0003).
    `iron_loss`: "steinmetz" (default), "lut", or None. Pass ``data_path`` to
    build the same drive from another machine's flux map."""
    return FluxMapDrive.from_npz(
        data_path, n_phases=5, n_ppairs=3, R_s=0.02194121,
        curr_max=curr_max, volt_max=volt_max,
        iron_loss=iron_loss, iron_k=(_TESLA5F_IRON_K1, _TESLA5F_IRON_K3),
        flux_backend=flux_backend,
    )


def im5_tesla_gen1_fluxmap_v2(
    curr_max: float = 404.0,
    volt_max: float = 230.0,
    iron_loss: str | None = "lut",
    flux_backend: str = "smooth",
) -> FluxMapDrive:
    """Same machine as :func:`im5_tesla_gen1_fluxmap` on the 2026-09 full FEM
    sweep (2127 points: |i_1| to 470 A, i_d3 0..80 A plus the old negative-i_d3
    points, i_q3 -60..80 A) with the rotor-cage loss column. Defaults: 2x rated
    peak current (rated 143 A rms = 202 A peak, design sheet e5_eMobility_IM_150kW),
    V_DC = 460 V (design sheet 461.8 V)."""
    return FluxMapDrive.from_npz(
        _DATA_V2, n_phases=5, n_ppairs=3, R_s=0.02194121,
        curr_max=curr_max, volt_max=volt_max,
        iron_loss=iron_loss, iron_k=(_TESLA5F_IRON_K1, _TESLA5F_IRON_K3),
        flux_backend=flux_backend,
    )


# 5-phase PMSM prototype, measured on the bench (not a design): R_s = 0.45 Ohm at
# 20 C from the bench import script, scaled with the copper temperature
# coefficient; the flux map's mean winding temperature is the default.
_PMSM5_R20 = 0.45  # Ohm
_ALPHA_CU = 0.00393  # 1/K
_DATA_DIR = Path(__file__).parent / "data"
# measurement campaigns: map file, physics-model file, converter limits of that study
_PMSM5_CAMPAIGNS = {
    # Oct 2026: 960-point grid; V_max = U_dc/2 at the sagging full-current DC link
    # (360 V), I_max assumed; the study used zero-sequence injection.
    "2026-10": dict(data="pmsm5_prototype_fluxmap.npz", sat="pmsm5_prototype_satmodel.json",
                    curr_max=50.0, volt_max=180.0),
    # Nov 2026 (Regatron DC source): v2 grid, 1520 points; limits U_dc = 350 V
    # (phase peak 175 V, no zero-sequence injection) and I_max = 45 A (vsmidl).
    "2026-11": dict(data="pmsm5_prototype_fluxmap_2026-11.npz", sat="pmsm5_prototype_satmodel_2026-11.json",
                    curr_max=45.0, volt_max=175.0),
}
_PMSM5_DEFAULT = "2026-11"
_DATA_PMSM5 = _DATA_DIR / _PMSM5_CAMPAIGNS[_PMSM5_DEFAULT]["data"]


def _pmsm5_setup(
    campaign: str, curr_max: float | None, volt_max: float | None, t_winding: float | None
) -> tuple[Path, float, float, float]:
    """map file, curr_max, volt_max, R_s for a campaign (None -> campaign default;
    t_winding default = the map's mean winding temperature)"""
    c = _PMSM5_CAMPAIGNS[campaign]
    path = _DATA_DIR / c["data"]
    if t_winding is None:
        t_winding = float(np.mean(np.load(path)["t_wdg"]))
    return (
        path,
        c["curr_max"] if curr_max is None else curr_max,
        c["volt_max"] if volt_max is None else volt_max,
        _PMSM5_R20 * (1.0 + _ALPHA_CU * (t_winding - 20.0)),
    )


def pmsm5_prototype_fluxmap(
    curr_max: float | None = None,
    volt_max: float | None = None,
    t_winding: float | None = None,
    flux_backend: str = "smooth",
    flux_smoothing: float = 0.0,
    campaign: str = _PMSM5_DEFAULT,
) -> FluxMapDrive:
    """Five-phase PMSM prototype (p_p = 2) as a direct MEASURED flux-map drive.

    Data: Laksar bench; each setpoint the average of +2000 and -2000 rpm (cancels
    the remaining inverter-voltage and resistance errors and drag); arrays
    currents, flux, T_meas, T_drag, udc, t_wdg. Valid for the hot machine in the
    motoring quadrant (i_d1 <= 0, i_q1 >= 0) -- use HullConstrainedOptimizer to
    stay there. No iron-loss model: the map has a single speed.

    campaign "2026-11" (default): 1516 setpoints of the v2 grid; flux from the
    APPLIED voltage (delay and per-phase inverter drop removed); T_meas is the
    shaft torque recalibrated to the flux torque (raw shaft torque 5.5 % above
    the flux torque, of which ~2.4 % is an artefact: the bench's averaging window
    lies in the dyno speed recovery after each step and the transducer reads
    T_em - J dw/dt, J ~ 0.021 kg m^2; ~3 % remain, sensor-gain-like; see
    2026-5f-pmsm/checks c14). Limits 175 V phase peak (U_dc 350 V, no zero-sequence
    injection: ``ForwardModel(add_volt_0=False)``) and 45 A; the controller must
    also command the inverter drop: ``ForwardModel(volt_drop=pmsm5_prototype_inverter())``.
    campaign "2026-10": 959 setpoints, flux from the commanded voltage; 180 V
    with zero-sequence injection, 50 A assumed."""
    path, i_max, v_max, R_s = _pmsm5_setup(campaign, curr_max, volt_max, t_winding)
    return FluxMapDrive.from_npz(
        path, n_phases=5, n_ppairs=2, R_s=R_s,
        curr_max=i_max, volt_max=v_max, iron_loss=None,
        flux_backend=flux_backend, symmetry="none", anchor_origin=False,
        seed_mode="data", flux_smoothing=flux_smoothing,
    )


def pmsm5_prototype_inverter() -> InverterDrop:
    """The prototype's inverter drop as fitted on the bench (Nov 2026 block means,
    +/- flux difference; R_sw on top of R_s at the winding temperature)."""
    U_dt, I_r, R_sw, _ = np.load(_DATA_DIR / _PMSM5_CAMPAIGNS["2026-11"]["data"])["vmodel"]
    return InverterDrop(U_dt=U_dt, I_r=I_r, R_sw=R_sw)
