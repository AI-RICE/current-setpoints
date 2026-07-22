from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from scipy.special import erf as _np_erf, expit as _np_sigmoid


_NP_ACTIVATIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "relu": lambda z: np.maximum(z, 0.0),
    "gelu": lambda z: 0.5 * z * (1.0 + _np_erf(z / math.sqrt(2.0))),
    "silu": lambda z: z * _np_sigmoid(z),
    "tanh": np.tanh,
    "leaky_relu": lambda z: np.where(z > 0, z, 0.01 * z),
}
_NP_ACTIVATION_DERIVATIVES: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "relu": lambda z: (z > 0).astype(z.dtype),
    "gelu": lambda z: 0.5 * (1.0 + _np_erf(z / math.sqrt(2.0)))
    + z * np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi),
    "silu": lambda z: _np_sigmoid(z) * (1.0 + z * (1.0 - _np_sigmoid(z))),
    "tanh": lambda z: 1.0 - np.tanh(z) ** 2,
    "leaky_relu": lambda z: np.where(z > 0, 1.0, 0.01),
}
# Second derivatives -- needed for NeuralFlux's Hessian-based inductance (see
# below): a single-hidden-layer MLP's Hessian w.r.t. its input is
# W1.T @ diag(w2 * act''(z)) @ W1, symmetric by construction regardless of
# the trained weights, which is exactly what guarantees a symmetric
# inductance matrix (Maxwell reciprocity) without having to hope training
# happens to produce one. relu/leaky_relu are piecewise-linear (zero
# second derivative a.e.; the kink at 0 is measure-zero and ignored).
_NP_ACTIVATION_SECOND_DERIVATIVES: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "relu": lambda z: np.zeros_like(z),
    "gelu": lambda z: (2.0 - z * z) * np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi),
    "silu": lambda z: _np_sigmoid(z) * (1.0 - _np_sigmoid(z)) * (2.0 + z * (1.0 - 2.0 * _np_sigmoid(z))),
    "tanh": lambda z: -2.0 * np.tanh(z) * (1.0 - np.tanh(z) ** 2),
    "leaky_relu": lambda z: np.zeros_like(z),
}


def _build_cross_coupling(n_harmonics: int) -> np.ndarray:
    dim = 2 * n_harmonics
    J = np.zeros((dim, dim))
    for i in range(n_harmonics):
        h = 2 * i + 1
        J[2 * i, 2 * i + 1] = -h
        J[2 * i + 1, 2 * i] = h
    return J


class FluxModel(ABC):
    cross_coupling: np.ndarray  # J — anti-symmetric rotation/coupling matrix

    @abstractmethod
    def flux(self, omega: float, curr_dq: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def inductance(self, omega: float, curr_dq: np.ndarray | None = None) -> np.ndarray:
        pass


class ConstantFlux(FluxModel):
    def __init__(
        self,
        flux_pm: np.ndarray,
        L_stat: np.ndarray,
        cross_coupling: np.ndarray,
    ) -> None:
        self.flux_pm = np.asarray(flux_pm)
        self.L_stat = np.asarray(L_stat)
        self.cross_coupling = np.asarray(cross_coupling)

    def flux(self, omega: float, curr_dq: np.ndarray) -> np.ndarray:
        return self.flux_pm

    def inductance(self, omega: float, curr_dq: np.ndarray | None = None) -> np.ndarray:
        return self.L_stat


class NeuralFlux(FluxModel):
    """Neural correction to the analytic ConstantFlux model, not a from-
    scratch flux predictor. The network outputs a single SCALAR magnetic
    co-energy residual W_res(omega, i); flux and inductance are then its
    gradient and Hessian w.r.t. i, ADDED to the analytic baseline:

        flux(omega, i)       = flux_pm + L_stat @ i + d(W_res)/di
        inductance(omega, i) = L_stat + d^2(W_res)/di^2

    Two deliberate properties, both by construction rather than by hoping
    training gets there:
      - inductance is symmetric: a Hessian of a scalar function is always
        symmetric (W1.T @ diag(w2 * act''(z)) @ W1 is symmetric regardless
        of the trained weights), unlike differentiating a raw 4-vector
        network output, which has no such guarantee.
      - flux/inductance degrade gracefully far outside the training data:
        with the network's own contribution near zero (e.g. if training
        regularizes the residual toward zero away from data, or simply by
        the network's limited capacity to extrapolate strongly), both
        reduce to the analytic ConstantFlux values -- which are well-
        behaved everywhere -- rather than to an ungrounded raw MLP output.
        This mirrors NeuralTorqueResidual's existing additive-correction
        pattern (see machines.py's neural_pmsm5phase), which never showed
        the same failure mode this replaces (see FAULT_TOLERANT.md).

    See notebooks/flux_nn_trainer.ipynb ("co-energy residual" section) for
    how the shipped weights were trained to be consistent with this.
    """

    def __init__(
        self,
        net: Any,  # NeuralFluxPredictor, output_size == 1 (scalar co-energy residual)
        scaler: Any,  # sklearn StandardScaler
        device: torch.device,
        L_stat: np.ndarray,
        cross_coupling: np.ndarray,
        flux_pm: np.ndarray,
    ) -> None:
        self.net = net
        self.scaler = scaler
        self.device = device
        self.L_stat = np.asarray(L_stat)
        self.cross_coupling = np.asarray(cross_coupling)
        self.flux_pm = np.asarray(flux_pm)

        with torch.no_grad():
            self._W1 = net.fc1.weight.cpu().numpy()  # (hidden, 5)
            self._b1 = net.fc1.bias.cpu().numpy()  # (hidden,)
            self._w2 = net.fc2.weight.cpu().numpy().reshape(-1)  # (hidden,) -- output_size == 1
            self._b2 = float(net.fc2.bias.cpu().numpy().reshape(-1)[0])
        self._mean = scaler.mean_
        self._scale = scaler.scale_
        self._act_deriv = _NP_ACTIVATION_DERIVATIVES[net.activation]
        self._act_second_deriv = _NP_ACTIVATION_SECOND_DERIVATIVES[net.activation]

        self._flux_cache: tuple[tuple[float, ...], np.ndarray] | None = None
        self._inductance_cache: tuple[tuple[float, ...], np.ndarray] | None = None

    def flux(self, omega: float, curr_dq: np.ndarray) -> np.ndarray:
        key = (float(omega), *(float(v) for v in curr_dq))
        if self._flux_cache is not None and self._flux_cache[0] == key:
            return self._flux_cache[1]
        x = np.hstack(([omega], curr_dq))
        x_normed = (x - self._mean) / self._scale
        z = self._W1 @ x_normed + self._b1  # (hidden,)
        d_act_dz = self._act_deriv(z)  # (hidden,)
        grad_normed = (self._w2 * d_act_dz) @ self._W1  # (5,) -- d(W_res)/d(x_normed)
        grad_x = grad_normed / self._scale  # d(W_res)/d(x_raw)
        d_wres_d_i = grad_x[1:]  # (dim,) -- drop the omega component
        result = self.flux_pm + self.L_stat @ curr_dq + d_wres_d_i
        self._flux_cache = (key, result)
        return result

    def inductance(self, omega: float, curr_dq: np.ndarray | None = None) -> np.ndarray:
        """L = L_stat + d^2(W_res)/di^2, the Hessian of the network's scalar
        co-energy residual w.r.t. current -- symmetric by construction (see
        class docstring), computed directly from the network's weights via
        W1.T @ diag(w2 * act''(z)) @ W1, exact (not an approximation)."""
        if curr_dq is None:
            curr_dq = np.zeros(self.L_stat.shape[0])
        key = (float(omega), *(float(v) for v in curr_dq))
        if self._inductance_cache is not None and self._inductance_cache[0] == key:
            return self._inductance_cache[1]
        x = np.hstack(([omega], curr_dq))
        x_normed = (x - self._mean) / self._scale
        z = self._W1 @ x_normed + self._b1  # (hidden,)
        d2_act_dz2 = self._act_second_deriv(z)  # (hidden,)
        weighted_W1 = (self._w2 * d2_act_dz2)[:, None] * self._W1  # (hidden, 5)
        hess_normed = self._W1.T @ weighted_W1  # (5, 5), symmetric by construction
        hess_x = hess_normed / np.outer(self._scale, self._scale)
        hess_i = hess_x[1:, 1:]  # (dim, dim)
        result = self.L_stat + hess_i
        self._inductance_cache = (key, result)
        return result


class NeuralTorqueResidual:
    """Callable torque-residual component: composed into a drive via its
    optional ``torque_residual`` constructor argument, mirroring how
    ``NeuralFlux`` composes into the ``flux`` argument."""

    def __init__(
        self,
        net: Any,  # NeuralTorquePredictor
        scaler: Any,  # sklearn StandardScaler
        device: torch.device,
    ) -> None:
        self.net = net
        self.scaler = scaler
        self.device = device

    def __call__(self, omega: float, curr_dq: np.ndarray) -> float:
        X = self.scaler.transform(np.hstack(([omega], curr_dq)).reshape(1, -1))
        with torch.no_grad():
            residual = self.net(torch.from_numpy(X).float().to(self.device))
        return float(residual.cpu().numpy().item())


class DriveModel(ABC):
    # identity
    n_phases: int
    n_harmonics: int
    dim: int  # = 2 * n_harmonics
    n_ppairs: int
    k_phase: float  # = n_phases / 2

    # converter limits. Deliberately NOT here: the speed horizon of a
    # setpoint map (opts["omega_max"] at grid time, see optimization/grid.py)
    # -- that is a property of the optimization run, not of the drive (an IM
    # has no electromagnetic speed ceiling; even for a PMSM the map horizon
    # is a run choice).
    curr_max: float
    volt_max: float

    # electrical
    R_stat: np.ndarray  # (dim, dim) stator resistance in dq frame
    k_v: float  # eddy-type iron-loss coefficient (ModelLossesSubstitution1)
    k_h: float  # hysteresis-type iron-loss coefficient

    @abstractmethod
    def voltage_operator(self, omega: float, curr_dq: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def bemf_dq(self, omega: float, curr_dq: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def inductance(self, omega: float, curr_dq: np.ndarray | None = None) -> np.ndarray:
        pass

    # mechanical

    @abstractmethod
    def torque(self, omega: float, curr_dq: np.ndarray) -> float:
        pass

    @abstractmethod
    def seeds(self, guess: np.ndarray | None = None) -> list[np.ndarray]:
        pass

    def torque_quadratic(self, omega: float) -> tuple[np.ndarray, np.ndarray, float]:
        dim = self.dim
        _state = np.random.get_state()
        np.random.seed(0)
        Xs = np.random.normal(scale=8.0, size=(300, dim))
        np.random.set_state(_state)
        idx = [(i, j) for i in range(dim) for j in range(i, dim)]
        feats = np.array([[x[i] * x[j] for (i, j) in idx] + list(x) + [1.0] for x in Xs])
        y = np.array([self.torque(omega, x) for x in Xs])
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

    def iron_loss(self, omega: float, volt_dq: np.ndarray) -> float:
        from ..utils.loss_fit import substitution_loss_features  # lazy — avoids circular import

        f_v, f_h = substitution_loss_features(omega, volt_dq, self.n_ppairs)
        return float(self.k_v * f_v + self.k_h * f_h)


@dataclass
class PMSMParams:
    """Data for one concrete 5-phase PMSM prototype — everything a
    ``PMSMDrive`` needs that isn't structural physics. Deliberately NOT here:
    the speed horizon of a setpoint map — see ``DriveModel``."""

    n_phases: int
    n_ppairs: int
    R_stat: np.ndarray
    L_stat: np.ndarray
    flux_pm: np.ndarray
    curr_max: float
    volt_max: float
    k_v: float = 0.0
    k_h: float = 0.0


# Fixed, reproducible sample of the (dim=4) current hypercube [-1,1]^4, used
# as additional SLSQP restart directions in PMSMDrive.seeds() (scaled by
# curr_max there). Generated once via a fixed-seed RNG so results stay
# deterministic across runs -- NOT reseeded per call, which would just
# repeat the same directions rather than exploring new ones.
_RANDOM_SEED_DIRECTIONS = np.random.default_rng(20260721).uniform(-1.0, 1.0, size=(8, 4))


class PMSMDrive(DriveModel):
    """Parameter-free PMSM physics. Concrete machines are built via a named
    factory (e.g. ``ieee_machine2()``) returning a configured instance;
    variation between machine variants is composition (``flux=``,
    ``torque_residual=``), not subclassing."""

    def __init__(
        self,
        params: PMSMParams,
        flux: FluxModel | None = None,
        torque_residual: Callable[[float, np.ndarray], float] | None = None,
    ) -> None:
        self.n_phases = params.n_phases
        self.n_harmonics = 2
        self.dim = 2 * self.n_harmonics
        self.n_ppairs = params.n_ppairs
        self.k_phase = self.n_phases / 2

        self.R_stat = params.R_stat
        self.curr_max = params.curr_max
        self.volt_max = params.volt_max
        self.k_v = params.k_v
        self.k_h = params.k_h

        cross_coupling = _build_cross_coupling(self.n_harmonics)
        self.flux: FluxModel = flux if flux is not None else ConstantFlux(params.flux_pm, params.L_stat, cross_coupling)
        self._torque_residual = torque_residual

    def voltage_operator(self, omega: float, curr_dq: np.ndarray) -> np.ndarray:
        L = self.flux.inductance(omega, curr_dq)
        return self.R_stat + omega * self.flux.cross_coupling @ L

    def bemf_dq(self, omega: float, curr_dq: np.ndarray) -> np.ndarray:
        flux_pm = self.flux.flux(omega, curr_dq)
        return omega * self.flux.cross_coupling @ flux_pm

    def inductance(self, omega: float, curr_dq: np.ndarray | None = None) -> np.ndarray:
        return self.flux.inductance(omega, curr_dq)

    def torque(self, omega: float, curr_dq: np.ndarray) -> float:
        flux_pm = self.flux.flux(omega, curr_dq)
        L = self.flux.inductance(omega, curr_dq)
        J = self.flux.cross_coupling
        k = self.n_phases * self.n_ppairs / 4.0
        A = k * (J @ L + L @ J.T)
        b = k * (J @ flux_pm)
        t = float(curr_dq @ A @ curr_dq + 2 * b @ curr_dq)
        if self._torque_residual is not None:
            t += self._torque_residual(omega, curr_dq)
        return t

    def seeds(self, guess: np.ndarray | None = None) -> list[np.ndarray]:
        dim = self.dim
        I = self.curr_max
        candidates: list[np.ndarray] = []
        default = np.zeros(dim)
        default[0] = 1.0
        candidates.append(default)
        if guess is not None:
            candidates.append(guess.copy())
        g_mtpa = np.zeros(dim)
        g_mtpa[1] = I * 0.95
        candidates.append(g_mtpa)
        g_fw = np.zeros(dim)
        g_fw[0] = -I * 0.9
        g_fw[1] = I * 0.1
        candidates.append(g_fw)

        # The four seeds above all have id3=iq3=0 -- blind to any operating
        # point whose torque optimum genuinely needs third-harmonic current
        # (common under fault, and for machines with real 3rd-harmonic
        # torque content) and to field-weakening deeper than -0.9*curr_max.
        if dim >= 4:
            for sign1, sign3 in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                c = np.zeros(dim)
                c[1] = sign1 * I * 0.5
                c[3] = sign3 * I * 0.5
                candidates.append(c)
        for frac in (0.5, 0.7, 0.99):
            c = np.zeros(dim)
            c[0] = -I * frac
            c[1] = I * (1.0 - frac) * 0.5 + I * 0.05
            candidates.append(c)

        # Hand-picked directions above still leave real gaps: verified
        # empirically that fixing one grid node's bad local optimum this way
        # just exposes a different node as the new bottleneck, at a
        # comparably bad value (down to a spurious negative "success" once,
        # on the fault-tolerant, high-speed field-weakening corner). Broad
        # random coverage is what actually closes the gap -- a cheap
        # pre-screen to avoid paying for every random seed's full SLSQP
        # solve was tried and does NOT work reliably (see optimizer._run_
        # slsqp's docstring: the seed that actually converges to the true
        # optimum can rank outside any cheap heuristic's top candidates
        # before ever being polished), so every one of these is genuinely
        # polished by the caller. Neither SLSQP tolerance (ftol down to
        # 1e-12) nor finite-difference step size (eps from 1e-10 to 1e-3)
        # changes the outcome at all from a fixed starting point -- this is
        # a genuine basin-of-attraction problem, not a numerical-precision
        # one, so more restarts (not tighter tolerances) is the only lever
        # that actually helps. 8 rows (2x the 4 empirically found sufficient
        # to fix the two known bottleneck nodes, as a margin for other
        # operating points) -- going to 16 gave identical results on the
        # cases tested, i.e. pure added cost.
        if dim >= 4:
            for row in _RANDOM_SEED_DIRECTIONS:
                candidates.append(row * I)
        return candidates


def ieee_machine2_params(*, curr_max: float = 30.0, volt_max: float = 13.0) -> PMSMParams:
    """Parameters for the 5-phase IEEE-Machine-2 prototype. R_stat, L_stat,
    flux_pm identified by ``utils.flux_fit.fit_pm_flux`` via joint least
    squares against measured voltage and torque in
    ``data/aggregated_file_means.csv``. ``curr_max``/``volt_max`` are
    converter limits, not machine physics, but travel with the params
    instance so ``PMSMDrive`` is fully constructed in one call; the
    setpoint-map speed horizon is a run parameter, not here (see
    ``optimization.grid.calculate_grid``'s ``opts["omega_max"]``)."""
    # As directly identified, this matrix is NOT symmetric (max asymmetry
    # ~86% of its own largest entry -- verified 2026-07-21) -- but a real
    # inductance matrix must be symmetric (energy reciprocity, not a
    # modeling choice). Symmetrized below; the raw identified matrix is
    # kept here for provenance.
    L_stat_raw = 1e-3 * np.array(
        [
            [0.0920, -0.0286, -0.0141, 0.0010],
            [-0.0133, 0.1090, -0.0008, -0.0092],
            [-0.0088, 0.0037, 0.0725, -0.0466],
            [-0.0041, -0.0053, 0.0475, 0.0722],
        ]
    )
    return PMSMParams(
        n_phases=5,
        n_ppairs=8,
        R_stat=np.diag([0.0191, 0.0514, 0.0805, 0.0801]),
        L_stat=0.5 * (L_stat_raw + L_stat_raw.T),
        flux_pm=np.array([1.13438169e-02, 1.71999345e-03, 1.57771228e-05, 1.56271556e-05]),
        curr_max=curr_max,
        volt_max=volt_max,
    )


def ieee_machine2(*, curr_max: float = 30.0, volt_max: float = 13.0) -> PMSMDrive:
    """5-phase IEEE-Machine-2 prototype."""
    return PMSMDrive(ieee_machine2_params(curr_max=curr_max, volt_max=volt_max))


def neural_pmsm5phase(
    net: Any,
    scaler: Any,
    device: torch.device,
    *,
    curr_max: float = 30.0,
    volt_max: float = 13.0,
) -> PMSMDrive:
    """IEEE-Machine-2 with a neural torque residual composed in."""
    return PMSMDrive(
        ieee_machine2_params(curr_max=curr_max, volt_max=volt_max),
        torque_residual=NeuralTorqueResidual(net, scaler, device),
    )


# Trained flux-network artifact shipped in weights/ (see notebooks/flux_nn_trainer.ipynb
# "co-energy residual" section for how it was produced): a single-hidden-layer
# MLP, hidden_size=24, GELU, input (omega, i_d1, i_q1, i_d3, i_q3) -> output a
# SCALAR magnetic co-energy residual (dim=1) -- NeuralFlux takes its gradient
# and Hessian w.r.t. current, added to the analytic ConstantFlux baseline
# (flux_pm, L_stat), rather than treating the network's output as the whole
# flux vector from scratch (see NeuralFlux's docstring for why).
_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights"
_FLUX_NN_HIDDEN_SIZE = 24
_FLUX_NN_ACTIVATION = "gelu"
_FLUX_NN_INPUT_SIZE = 5
_FLUX_NN_OUTPUT_SIZE = 1


def ieee_machine2_trained_neural_flux(
    *,
    device: torch.device | None = None,
    curr_max: float = 30.0,
    volt_max: float = 13.0,
) -> PMSMDrive:
    """IEEE-Machine-2 with the trained neural flux co-energy residual from
    ``weights/`` (``FluxNN_Weights.pth``/``FluxNN_Scaler.npy``) composed in as
    a correction on top of the analytic ``ConstantFlux`` baseline (see
    ``NeuralFlux``)."""
    from ..utils.neural_model import load_neural_flux_model  # lazy — avoids circular import

    device = device if device is not None else torch.device("cpu")
    net, scaler = load_neural_flux_model(
        str(_WEIGHTS_DIR / "FluxNN_Weights.pth"),
        str(_WEIGHTS_DIR / "FluxNN_Scaler.npy"),
        hidden_size=_FLUX_NN_HIDDEN_SIZE,
        input_size=_FLUX_NN_INPUT_SIZE,
        output_size=_FLUX_NN_OUTPUT_SIZE,
        device=device,
        activation=_FLUX_NN_ACTIVATION,
    )
    params = ieee_machine2_params(curr_max=curr_max, volt_max=volt_max)
    flux = NeuralFlux(net, scaler, device, params.L_stat, _build_cross_coupling(2), params.flux_pm)
    return PMSMDrive(params, flux=flux)


@dataclass
class IMParams:
    """Data for one concrete 9-phase induction-motor prototype. Deliberately
    NOT here: the speed horizon of a setpoint map — see ``DriveModel``."""

    n_phases: int
    n_ppairs: int
    R_s: float
    R_r: np.ndarray
    L_mu: np.ndarray
    L_s_sigma: np.ndarray
    L_r_sigma: np.ndarray
    curr_max: float
    volt_max: float
    k_v: float = 0.0
    k_h: float = 0.0


class InductionDrive(DriveModel):
    """Parameter-free induction-motor physics. Concrete machines are built
    via a named factory (e.g. ``im9_prototype()``)."""

    def __init__(self, params: IMParams) -> None:
        self.n_phases = params.n_phases
        self.n_harmonics = 2
        self.dim = 2 * self.n_harmonics
        self.n_ppairs = params.n_ppairs
        self.k_phase = self.n_phases / 2

        self.R_stat = params.R_s * np.eye(self.dim)
        self.curr_max = params.curr_max
        self.volt_max = params.volt_max
        self.k_v = params.k_v
        self.k_h = params.k_h

        self._R_r = params.R_r
        self._L_mu = params.L_mu
        self._L_s = self._L_mu + params.L_s_sigma
        self._L_r = self._L_mu + params.L_r_sigma
        self._cross_coupling = _build_cross_coupling(self.n_harmonics)

    def k_ir(self, omega_r: float) -> np.ndarray:
        K = np.zeros((self.dim, self.dim))
        for i in range(self.n_harmonics):
            h = 2 * i + 1
            L_mu_h = self._L_mu[2 * i, 2 * i]
            L_r_h = self._L_r[2 * i, 2 * i]
            R_r_h = self._R_r[2 * i, 2 * i]
            K[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = self._k_ir_block(omega_r, L_mu_h, L_r_h, R_r_h, h)
        return K

    @staticmethod
    def _k_ir_block(omega_r: float, L_mu_h: float, L_r_h: float, R_r_h: float, h: int) -> np.ndarray:
        h_omega_r = h * omega_r
        denom = R_r_h**2 + (h_omega_r * L_r_h) ** 2
        if denom == 0.0:
            return np.zeros((2, 2))
        scale = h_omega_r * L_mu_h / denom
        return scale * np.array(
            [
                [-h_omega_r * L_r_h, R_r_h],
                [-R_r_h, -h_omega_r * L_r_h],
            ]
        )

    def slip_from_dq_foc(self, curr_dq: np.ndarray, eps: float = 1e-9) -> float:
        i_sd_1 = curr_dq[0]
        i_sq_1 = curr_dq[1]
        if abs(i_sq_1) < eps:
            return 0.0
        return (self._R_r[0, 0] / self._L_r[0, 0]) * (i_sd_1 / i_sq_1)

    def voltage_operator(self, omega: float, curr_dq: np.ndarray) -> np.ndarray:
        return self.R_stat + omega * self._cross_coupling @ self.inductance(omega, curr_dq)

    def bemf_dq(self, omega: float, curr_dq: np.ndarray) -> np.ndarray:
        return np.zeros(self.dim)

    def inductance(self, omega: float, curr_dq: np.ndarray | None = None) -> np.ndarray:
        if curr_dq is None:
            return self._L_s
        omega_r = self.slip_from_dq_foc(curr_dq)
        return self._L_s + self._L_mu @ self.k_ir(omega_r)

    def torque(self, omega: float, curr_dq: np.ndarray) -> float:
        omega_r = self.slip_from_dq_foc(curr_dq)
        A = (self.n_phases * self.n_ppairs / 2.0) * (self._cross_coupling @ self._L_mu @ self.k_ir(omega_r))
        return float(curr_dq @ A @ curr_dq)

    def seeds(self, guess: np.ndarray | None = None) -> list[np.ndarray]:
        dim = self.dim
        I = self.curr_max
        candidates: list[np.ndarray] = []
        if guess is not None:
            candidates.append(guess.copy())
        c = np.zeros(dim)
        c[0] = 0.1
        c[1] = 0.1
        candidates.append(c)
        for amp in (0.3, 0.5, 0.7):
            c = np.zeros(dim)
            c[0] = I * amp
            c[1] = I * amp
            candidates.append(c)
        c = np.zeros(dim)
        c[0] = I * 0.85
        c[1] = I * 0.30
        candidates.append(c)
        if dim >= 4:
            c = np.zeros(dim)
            c[0] = I * 0.7
            c[1] = I * 0.7
            c[2] = I * 0.1
            c[3] = -I * 0.1
            candidates.append(c)
        return candidates

    def copper_loss(self, omega: float, curr_dq: np.ndarray) -> float:
        R_s = self.R_stat[0, 0]
        P_stator = self.k_phase * R_s * float(curr_dq @ curr_dq)
        omega_r = self.slip_from_dq_foc(curr_dq)
        i_r = self.k_ir(omega_r) @ curr_dq
        P_rotor = self.k_phase * float(i_r @ self._R_r @ i_r)
        return P_stator + P_rotor


def im9_prototype_params(*, curr_max: float = 20.0, volt_max: float = 200.0) -> IMParams:
    """Parameters for the 9-phase induction-motor prototype. The setpoint-map
    speed horizon is a run parameter, not here (see
    ``optimization.grid.calculate_grid``'s ``opts["omega_max"]``) -- an IM
    has no electromagnetic speed ceiling in the first place."""
    return IMParams(
        n_phases=9,
        n_ppairs=2,
        R_s=5.0,
        R_r=np.diag([1.54, 1.54, 1.57, 1.57]),
        L_mu=1e-3 * np.diag([496.0, 496.0, 58.2, 58.2]),
        L_s_sigma=1e-3 * np.diag([15.1, 15.1, 13.6, 13.6]),
        L_r_sigma=1e-3 * np.diag([53.3, 53.3, 33.4, 33.4]),
        curr_max=curr_max,
        volt_max=volt_max,
    )


def im9_prototype(*, curr_max: float = 20.0, volt_max: float = 200.0) -> InductionDrive:
    return InductionDrive(im9_prototype_params(curr_max=curr_max, volt_max=volt_max))
