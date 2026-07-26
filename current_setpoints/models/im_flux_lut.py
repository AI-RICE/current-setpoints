"""High-fidelity flux-linkage lookup model for the Tesla1-class five-phase IM.

Where `im_lut.im5_tesla_gen1` is an analytic equivalent circuit with an
identified 1-D saturation law (ADR 0003, ~11 Nm torque RMS), this model
interpolates the raw Ansys flux-linkage map psi(i) directly. It reproduces the
357-point FEM torque sweep to ~0.6 Nm RMS (0.5 %) because it carries the full
cross-saturation and h=1<->h=3 coupling that no 1-D law can.

Torque is the co-energy expression T = (m*p_p/2) * i . J . psi, with J the
codebase's cross-coupling matrix (which embeds the harmonic weight h, so the
3rd-harmonic term is automatically weighted by 3). Terminal voltage in the
synchronous frame follows from the same map: v = R_s*i + omega * J . psi.

Scope / limitations:
  * Valid inside the convex hull of the sampled currents (Id1 40..160,
    Iq1 0..200, |I3| <= 40 A). Outside, `flux` falls back to nearest-neighbour
    (flagged by `in_hull`) — do not trust extrapolated values.
  * The map is *stator* flux linkage vs *stator* current: torque, terminal
    voltage and stator copper loss are exact; rotor current / rotor copper
    loss are NOT recoverable from it (use the equivalent-circuit model for
    loss decomposition). Iron loss is available separately as a FEM column but
    is not yet wired in.
  * Linear (Delaunay) interpolation has kinked gradients at simplex faces;
    fine for evaluation, but gradient-based optimisation over this model
    should be validated (splines/RBF are the fallback — see ADR 0003).
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import ConvexHull

from .machines import DriveModel, _build_cross_coupling

_DATA = Path(__file__).parent / "data" / "tesla5f_fluxmap.csv"

# Electrical fundamental frequency [rad/s] at which the Ansys sweep was run.
# UNCONFIRMED — the sweep carries no speed/frequency column and every point is
# averaged over one time window, so iron loss is characterised at a *single*
# speed only. Needed to (a) guard the LUT and (b) scale the Steinmetz model to
# other speeds. Ask Laksar; until then both stay pinned to the sweep speed.
_TESLA5F_FEM_OMEGA_EL: float | None = None

# Steinmetz-type iron-loss coefficients, P = k1|psi1|^2 + k3|psi3|^2, fit to
# the 357-point FEM CoreLoss column (no offset): R^2 = 0.994, train RMS 19 W
# (2.5%). k3 already folds the h=3 plane's 3x-frequency weighting. At the sweep
# speed this GENERALISES BETTER than interpolating the CoreLoss column directly
# (leave-one-out 2.5% vs 4.8%), because 4-D scattered-linear interpolation is
# noisier than the B^2 physical prior. See ADR 0003.
_TESLA5F_IRON_K1 = 4.75427e4  # W / Wb^2, fundamental plane
_TESLA5F_IRON_K3 = 1.353636e6  # W / Wb^2, third-harmonic plane


_MLP_WEIGHTS = Path(__file__).parent / "data" / "tesla5f_flux_mlp.npz"
_MLP_HIDDEN = (64, 64)  # architecture is fixed in code; weights live in the .npz


def _silu(x: np.ndarray) -> np.ndarray:
    # x * sigmoid(x), overflow-safe for large |x| (optimiser probes extrapolate)
    z = np.clip(x, -60.0, 60.0)
    return x / (1.0 + np.exp(-z))


class _FluxMLP:
    """Tiny MLP surrogate of the flux map (4 -> 64 -> 64 -> 4, SiLU) with
    input/output standardisation. Inference is pure numpy (no torch). It fits
    the 357 FEM points to ~0.6 Nm torque RMS *and* generalises (20% held-out
    ~0.6 Nm), while being C-infinity smooth — unlike the Delaunay interpolant,
    whose gradients kink at simplex faces. Odd symmetry psi(-i) = -psi(i) is
    imposed by the caller (IMFluxLUT._canon), so the net only needs the
    positive-Id1 half of the data. Retrain via `train_flux_mlp` when a new FEM
    sweep (new xlsx -> new CSV) arrives; normal use loads the committed .npz."""

    def __init__(self, path: Path = _MLP_WEIGHTS) -> None:
        z = np.load(path)
        n = len(_MLP_HIDDEN) + 1
        self.W = [z[f"W{i}"] for i in range(n)]
        self.b = [z[f"b{i}"] for i in range(n)]
        self.xm, self.xs, self.ym, self.ys = z["xm"], z["xs"], z["ym"], z["ys"]

    def __call__(self, curr: np.ndarray) -> np.ndarray:
        x = (np.asarray(curr, float) - self.xm) / self.xs
        for W, b in zip(self.W[:-1], self.b[:-1]):
            x = _silu(x @ W + b)
        return (x @ self.W[-1] + self.b[-1]) * self.ys + self.ym


def train_flux_mlp(
    csv: Path = _DATA,
    out: Path = _MLP_WEIGHTS,
    seed: int = 0,
    epochs: int = 6000,
    lr: float = 2e-3,
    weight_decay: float = 1e-5,
) -> Path:
    """(Re)fit the flux-map MLP from `csv` and save weights to `out`. Run this
    ONLY when a new FEM sweep arrives (new xlsx -> regenerated CSV); normal use
    loads the committed weights. Deterministic for a given seed. Uses torch for
    training only; inference (`_FluxMLP`) is numpy."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    d = np.loadtxt(csv, delimiter=",", skiprows=1)
    X, Y = d[:, 0:4].astype(np.float32), d[:, 4:8].astype(np.float32)
    xm, xs, ym, ys = X.mean(0), X.std(0), Y.mean(0), Y.std(0)
    net = nn.Sequential(
        nn.Linear(4, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 4)
    )
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    xa = torch.tensor((X - xm) / xs)
    ya = torch.tensor((Y - ym) / ys)
    for _ in range(epochs):
        opt.zero_grad()
        (((net(xa) - ya) ** 2).mean()).backward()
        opt.step()
    lins = [m for m in net if isinstance(m, nn.Linear)]
    arrs = {}
    for i, m in enumerate(lins):
        arrs[f"W{i}"] = m.weight.detach().numpy().T.astype(np.float64)  # x @ W + b
        arrs[f"b{i}"] = m.bias.detach().numpy().astype(np.float64)
    np.savez(out, xm=xm, xs=xs, ym=ym, ys=ys, **arrs)
    return out


class IronLossModel(ABC):
    """Iron-loss strategy for a flux-map machine. Kept separate from the
    magnetics so either machine (analytic EC or flux LUT) can hold one, exactly
    as `FluxModel` is separated in `machines.py`."""

    @abstractmethod
    def loss(self, omega: float, curr_dq: np.ndarray, flux: np.ndarray) -> float:
        """Iron loss [W] at electrical speed `omega`, given the operating-point
        current and its flux linkage (both supplied so a strategy can key on
        whichever it needs)."""


class SteinmetzIronLoss(IronLossModel):
    """Physical loss P = k1|psi1|^2 + k3|psi3|^2 (loss ~ B^2 at fixed
    frequency). Default (`omega_ref=None`) returns the sweep-speed value. If a
    reference speed is set, scales by (omega/omega_ref)^freq_exp — an ASSUMED
    frequency law (eddy-like at freq_exp=2), UNVALIDATED because the FEM sweep
    is single-speed; treat off-reference-speed values as an estimate."""

    def __init__(
        self,
        k1: float = _TESLA5F_IRON_K1,
        k3: float = _TESLA5F_IRON_K3,
        omega_ref: float | None = _TESLA5F_FEM_OMEGA_EL,
        freq_exp: float = 2.0,
    ) -> None:
        self.k1, self.k3, self.omega_ref, self.freq_exp = k1, k3, omega_ref, freq_exp

    def loss(self, omega: float, curr_dq: np.ndarray, flux: np.ndarray) -> float:
        p = self.k1 * (flux[0] ** 2 + flux[1] ** 2) + self.k3 * (flux[2] ** 2 + flux[3] ** 2)
        if self.omega_ref:
            p *= (abs(omega) / self.omega_ref) ** self.freq_exp
        return float(p)


class CoreLossLUT(IronLossModel):
    """Interpolates the FEM CoreLoss column directly over the dq currents.
    Faithful *at the sweep speed* (leave-one-out RMS ~4.8%); does not model
    frequency, so it warns if queried far from `omega_ref` (when known)."""

    def __init__(
        self,
        currents: np.ndarray,
        p_core: np.ndarray,
        omega_ref: float | None = _TESLA5F_FEM_OMEGA_EL,
    ) -> None:
        self.omega_ref = omega_ref
        pts = np.vstack([np.zeros((1, 4)), np.asarray(currents, float)])
        val = np.concatenate([[0.0], np.asarray(p_core, float)])
        self._lin = LinearNDInterpolator(pts, val)
        self._near = NearestNDInterpolator(pts, val)

    def loss(self, omega: float, curr_dq: np.ndarray, flux: np.ndarray) -> float:
        if self.omega_ref and abs(abs(omega) - self.omega_ref) > 0.05 * self.omega_ref:
            warnings.warn(
                "CoreLossLUT queried away from the sweep speed; it does not "
                "model frequency — use SteinmetzIronLoss for other speeds.",
                stacklevel=2,
            )
        q = np.asarray(curr_dq, float).reshape(1, 4)
        p = self._lin(q)[0]
        if np.isnan(p):
            p = self._near(q)[0]
        return float(p)


class IMFluxLUT(DriveModel):
    """Induction drive whose magnetics are a scattered flux-linkage LUT.

    Args:
        currents: (N, 4) sampled dq currents [Id1, Iq1, Id3, Iq3].
        flux: (N, 4) stator flux linkages [psi_d1, psi_q1, psi_d3, psi_q3].
        n_phases, n_ppairs, R_s, curr_max, volt_max: machine/converter data.
    """

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
        flux_backend: str = "linear",
    ) -> None:
        self.n_phases = n_phases
        self.n_harmonics = 2
        self.dim = 4
        self.n_ppairs = n_ppairs
        self.k_phase = n_phases / 2
        self.iron_loss_model = iron_loss_model
        self.flux_backend = flux_backend
        self._mlp = _FluxMLP() if flux_backend == "mlp" else None

        self.R_stat = R_s * np.eye(self.dim)
        self.k_v = 0.0
        self.k_h = 0.0
        self.curr_max = curr_max
        self.volt_max = volt_max

        self._J = _build_cross_coupling(self.n_harmonics)
        # anchor the origin (psi = 0 at i = 0) so low-current queries stay
        # inside the hull; the FEM sweep itself starts at Id1 = 40 A.
        pts = np.vstack([np.zeros((1, 4)), np.asarray(currents, float)])
        val = np.vstack([np.zeros((1, 4)), np.asarray(flux, float)])
        self._lin = LinearNDInterpolator(pts, val)
        self._near = NearestNDInterpolator(pts, val)
        # facet inequalities A x + b <= 0 of the sampled-current convex hull,
        # for confining an optimiser to the FEM envelope (no extrapolation).
        self._hull_eq = ConvexHull(pts).equations  # (n_facets, dim + 1)

    @staticmethod
    def _canon(curr_dq: np.ndarray) -> tuple[float, np.ndarray]:
        """Sign-canonicalise to Id1 >= 0, exploiting the odd symmetry
        psi(-i) = -psi(i) (torque and loss are invariant under i -> -i). Maps
        the mirror-image operating points the optimiser sometimes returns back
        onto the sampled (positive-Id1) data."""
        c = np.asarray(curr_dq, float)
        return (-1.0, -c) if c[0] < 0 else (1.0, c)

    def flux(self, omega: float, curr_dq: np.ndarray) -> np.ndarray:
        """Stator flux linkage psi(i). MLP (smooth) or Delaunay-linear backend;
        odd symmetry applied. Linear backend falls back to nearest-neighbour
        outside the sampled hull (flagged by `in_hull`)."""
        s, cc = self._canon(curr_dq)
        if self._mlp is not None:
            return s * self._mlp(cc)
        q = cc.reshape(1, 4)
        psi = self._lin(q)[0]
        if np.any(np.isnan(psi)):
            psi = self._near(q)[0]
        return s * psi

    def in_hull(self, curr_dq: np.ndarray) -> bool:
        """True iff the (sign-canonicalised) point is inside the sampled hull,
        i.e. psi(i) is interpolation rather than extrapolation."""
        _, cc = self._canon(curr_dq)
        return not np.any(np.isnan(self._lin(cc.reshape(1, 4))[0]))

    def hull_margins(self, curr_dq: np.ndarray) -> np.ndarray:
        """Signed distances to each facet of the sampled-current convex hull;
        every entry >= 0 iff the point is inside. Use as an SLSQP inequality
        constraint (`HullConstrainedOptimizer`) so setpoints never require
        extrapolating the flux map. NOT sign-canonicalised: the hull lies in
        Id1 >= 0, so this also confines the solve to the sampled sign-half
        (equivalent by odd symmetry, no torque lost)."""
        x = np.asarray(curr_dq, float)
        return -(self._hull_eq[:, :-1] @ x + self._hull_eq[:, -1])

    def torque(self, omega: float, curr_dq: np.ndarray) -> float:
        psi = self.flux(omega, curr_dq)
        return float(self.n_phases * self.n_ppairs / 2.0 * curr_dq @ self._J @ psi)

    def bemf_dq(self, omega: float, curr_dq: np.ndarray) -> np.ndarray:
        # speed voltage from the flux map; J embeds the per-plane frequency h
        return omega * self._J @ self.flux(omega, curr_dq)

    def voltage_operator(self, omega: float, curr_dq: np.ndarray) -> np.ndarray:
        # v = R_stat @ i + bemf_dq; the flux-induced part lives in bemf_dq
        return self.R_stat

    def inductance(self, omega: float, curr_dq: np.ndarray | None = None) -> np.ndarray:
        """Per-plane secant inductance ||psi_h|| / ||i_h|| (diagnostic only;
        the voltage path uses `flux` directly, not this)."""
        if curr_dq is None:
            return np.zeros((self.dim, self.dim))
        psi = self.flux(omega, curr_dq)
        L = np.zeros((self.dim, self.dim))
        for i in range(self.n_harmonics):
            sl = slice(2 * i, 2 * i + 2)
            i_norm = float(np.hypot(curr_dq[2 * i], curr_dq[2 * i + 1]))
            if i_norm > 1e-9:
                L[sl, sl] = float(np.hypot(psi[2 * i], psi[2 * i + 1])) / i_norm * np.eye(2)
        return L

    def copper_loss(self, omega: float, curr_dq: np.ndarray) -> float:
        """Stator copper loss only (rotor current is not in the flux map)."""
        return self.k_phase * self.R_stat[0, 0] * float(curr_dq @ curr_dq)

    def iron_loss_at(self, omega: float, curr_dq: np.ndarray) -> float:
        """Iron loss [W] via the machine's `iron_loss_model`. Current-keyed
        (unlike the voltage-keyed base `DriveModel.iron_loss`) because this
        machine is defined by its current->flux map. Validated only at the FEM
        sweep speed; see the strategy classes for the speed caveat."""
        if self.iron_loss_model is None:
            raise ValueError("no iron_loss_model set on this machine")
        return self.iron_loss_model.loss(omega, curr_dq, self.flux(omega, curr_dq))

    def seeds(self, guess: np.ndarray | None = None) -> list[np.ndarray]:
        i_max = self.curr_max
        c: list[np.ndarray] = []
        if guess is not None:
            c.append(guess.copy())
        c.append(np.array([0.1, 0.1, 0.0, 0.0]))
        for amp in (0.3, 0.5, 0.7):
            c.append(np.array([i_max * amp, i_max * amp, 0.0, 0.0]))
        c.append(np.array([i_max * 0.85, i_max * 0.30, 0.0, 0.0]))
        c.append(np.array([i_max * 0.7, i_max * 0.7, i_max * 0.1, -i_max * 0.1]))
        return c


def im5_tesla_gen1_fluxlut(
    curr_max: float = 200.0,
    volt_max: float = 230.0,
    iron_loss: str | None = "steinmetz",
    flux_backend: str = "linear",
) -> IMFluxLUT:
    """Tesla1-class five-phase IM as a direct FEM flux-linkage LUT.

    Data: the 357-point Ansys sweep I_combs_Results_correct_wr_definition.xlsx
    (columns Flux_d1/q1/d3/q3, CoreLoss), archived as `data/tesla5f_fluxmap.csv`.
    m = 5, p_p = 3, R_s = 21.94 mOhm (the computed EC value; ADR 0003).
    Reproduces the FEM torque sweep to ~0.6 Nm RMS. See module docstring for
    scope limits.

    iron_loss selects the `IronLossModel`:
      "steinmetz" (default) — physical P = k1|psi1|^2 + k3|psi3|^2; generalises
          best (leave-one-out 2.5%) and is the only variant that extrapolates
          to other speeds (with the flagged frequency assumption);
      "lut" — interpolate the FEM CoreLoss column directly (~4.8% LOO), the
          raw at-speed reference;
      None — no iron-loss model (`iron_loss_at` then raises)."""
    d = np.loadtxt(_DATA, delimiter=",", skiprows=1)
    currents = d[:, 0:4]  # Id1, Iq1, Id3, Iq3
    flux = d[:, 4:8]  # psi_d1, psi_q1, psi_d3, psi_q3
    p_core = d[:, 9]  # W

    if iron_loss == "steinmetz":
        ilm: IronLossModel | None = SteinmetzIronLoss()
    elif iron_loss == "lut":
        ilm = CoreLossLUT(currents, p_core)
    elif iron_loss is None:
        ilm = None
    else:
        raise ValueError(f"iron_loss must be 'steinmetz', 'lut' or None, got {iron_loss!r}")

    return IMFluxLUT(
        currents=currents,
        flux=flux,
        n_phases=5,
        n_ppairs=3,
        R_s=0.02194121,
        curr_max=curr_max,
        volt_max=volt_max,
        iron_loss_model=ilm,
        flux_backend=flux_backend,
    )
