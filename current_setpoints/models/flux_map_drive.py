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
inside the FEM envelope (no extrapolation). Odd symmetry psi(-i) = -psi(i) is
imposed (`_canon`), so only the positive-Id1 half of the data is needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, RBFInterpolator
from scipy.spatial import ConvexHull

from .iron_loss import CoreLossLUT, IronLossModel, SteinmetzIronLoss
from .machines import DriveModel, _build_cross_coupling

_DATA = Path(__file__).parent / "data" / "tesla5f_fluxmap.npz"

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
    ) -> None:
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
        # origin-anchored so low-current queries stay inside the hull
        pts = np.vstack([np.zeros((1, 4)), np.asarray(currents, float)])
        val = np.vstack([np.zeros((1, 4)), np.asarray(flux, float)])
        self._hull_eq = ConvexHull(pts).equations
        if flux_backend == "smooth":
            # C-infinity thin-plate-spline: unbiased at the curved peak-torque
            # edge where Delaunay-linear reads ~5% low (see ADR 0003).
            self._rbf = RBFInterpolator(pts, val, kernel="thin_plate_spline")
        elif flux_backend == "linear":
            self._lin = LinearNDInterpolator(pts, val)
            self._near = NearestNDInterpolator(pts, val)
        else:
            raise ValueError(f"flux_backend must be 'smooth' or 'linear', got {flux_backend!r}")
        # a Delaunay membership test for `in_hull`, regardless of backend
        self._member = LinearNDInterpolator(pts, val[:, :1])

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
    ) -> "FluxMapDrive":
        """Build from an ``.npz`` flux map --- machine-agnostic, point it at any
        machine's data. Required arrays: ``currents`` [N,4] and ``flux`` [N,4]
        (dq of harmonics 1,3); ``P_core`` [N] is needed only for
        ``iron_loss="lut"``. ``iron_loss``: ``None``, ``"lut"`` (uses
        ``P_core``), or ``"steinmetz"`` (needs ``iron_k=(k1, k3)``)."""
        z = np.load(path)
        currents, flux = z["currents"], z["flux"]
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
            iron_loss_model=ilm, flux_backend=flux_backend,
        )

    @staticmethod
    def _canon(curr_dq: np.ndarray) -> tuple[float, np.ndarray]:
        c = np.asarray(curr_dq, float)
        return (-1.0, -c) if c[0] < 0 else (1.0, c)

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
        """Stator copper loss only (rotor current is not in a flux map)."""
        return self.k_phase * self.R_stat[0, 0] * float(curr_dq @ curr_dq)

    def iron_loss_at(self, omega: float, curr_dq: np.ndarray) -> float:
        if self.iron_loss_model is None:
            raise ValueError("no iron_loss_model set on this drive")
        return self.iron_loss_model.loss(omega, curr_dq, self.flux(omega, curr_dq))

    def seeds(self, guess: np.ndarray | None = None) -> list[np.ndarray]:
        i_max = self.curr_max
        c: list[np.ndarray] = []
        if guess is not None:
            c.append(np.asarray(guess, float).copy())
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
