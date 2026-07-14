from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from scipy.special import erf as _np_erf, expit as _np_sigmoid

# Plain-numpy activations/derivatives for utils.neural_model.ACTIVATIONS, used
# by NeuralFlux to evaluate its tiny (single-hidden-layer) network without
# torch's per-op dispatch overhead — large relative to a network this size
# called this many times during a grid/optimization sweep. Verified to match
# the torch forward pass and torch.autograd.functional.jacobian to float
# precision (~1e-9/1e-10).
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
    def __init__(
        self,
        net: Any,  # NeuralFluxPredictor
        scaler: Any,  # sklearn StandardScaler
        device: torch.device,
        L_stat: np.ndarray,
        cross_coupling: np.ndarray,
    ) -> None:
        self.net = net
        self.scaler = scaler
        self.device = device
        self.L_stat = np.asarray(L_stat)
        self.cross_coupling = np.asarray(cross_coupling)
        # Weights/scaler pulled once into plain numpy arrays: this network is
        # tiny (single hidden layer, ~24 units) and evaluated hundreds of
        # thousands of times per grid, so torch's per-op dispatch overhead
        # (present even under no_grad()) dominates over the actual FLOPs.
        # Forward pass and its analytic derivative are done in numpy instead
        # — verified to match the torch versions to float precision
        # (~1e-9/1e-10), not an approximation.
        with torch.no_grad():
            self._W1 = net.fc1.weight.cpu().numpy()
            self._b1 = net.fc1.bias.cpu().numpy()
            self._W2 = net.fc2.weight.cpu().numpy()
            self._b2 = net.fc2.bias.cpu().numpy()
        self._mean = scaler.mean_
        self._scale = scaler.scale_
        self._act = _NP_ACTIVATIONS[net.activation]
        self._act_deriv = _NP_ACTIVATION_DERIVATIVES[net.activation]
        # Last-call caches: the optimizer's constraint functions (current limit,
        # voltage limit, torque equality) each independently re-derive flux/
        # inductance for the SAME (omega, curr_dq) within one SLSQP evaluation
        # point. A single-slot cache turns that 3x redundancy into 1x without
        # touching the shared optimizer/forward-model code.
        self._flux_cache: tuple[tuple[float, ...], np.ndarray] | None = None
        self._inductance_cache: tuple[tuple[float, ...], np.ndarray] | None = None

    def flux(self, omega: float, curr_dq: np.ndarray) -> np.ndarray:
        key = (float(omega), *(float(v) for v in curr_dq))
        if self._flux_cache is not None and self._flux_cache[0] == key:
            return self._flux_cache[1]
        x = np.hstack(([omega], curr_dq))
        x_normed = (x - self._mean) / self._scale
        z = self._W1 @ x_normed + self._b1
        a = self._act(z)
        result = self._W2 @ a + self._b2
        self._flux_cache = (key, result)
        return result

    def inductance(self, omega: float, curr_dq: np.ndarray | None = None) -> np.ndarray:
        """
        L = L_stat + d(flux)/d(i), where the derivative is computed directly
        from the network's weights (single hidden layer: fc1 -> act -> fc2)
        via the chain rule: d(out)/d(x) = W2 @ diag(act'(z)) @ W1 / scale.
        Exact (not an approximation) — verified to match
        ``torch.autograd.functional.jacobian`` to float precision.
        """
        if curr_dq is None:
            curr_dq = np.zeros(self.L_stat.shape[0])
        key = (float(omega), *(float(v) for v in curr_dq))
        if self._inductance_cache is not None and self._inductance_cache[0] == key:
            return self._inductance_cache[1]
        x = np.hstack(([omega], curr_dq))
        x_normed = (x - self._mean) / self._scale
        z = self._W1 @ x_normed + self._b1
        d_act_dz = self._act_deriv(z)
        d_out_d_xnormed = self._W2 @ (d_act_dz[:, None] * self._W1)  # (dim, 1 + dim)
        d_out_d_x = d_out_d_xnormed / self._scale
        d_flux_d_i = d_out_d_x[:, 1:]
        result = self.L_stat + d_flux_d_i
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

    # limits
    curr_max: float
    volt_max: float
    omega_max: float

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

    def set_max_pars(self, curr_max: float, volt_max: float, omega_max: float) -> None:
        self.curr_max = curr_max
        self.volt_max = volt_max
        self.omega_max = omega_max


@dataclass
class PMSMParams:
    """Data for one concrete 5-phase PMSM prototype — everything a
    ``PMSMDrive`` needs that isn't structural physics."""

    n_phases: int
    n_ppairs: int
    R_stat: np.ndarray
    L_stat: np.ndarray
    flux_pm: np.ndarray
    k_v: float = 0.0
    k_h: float = 0.0


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
        candidates: list[np.ndarray] = []
        default = np.zeros(dim)
        default[0] = 1.0
        candidates.append(default)
        if guess is not None:
            candidates.append(guess.copy())
        g_mtpa = np.zeros(dim)
        g_mtpa[1] = self.curr_max * 0.95
        candidates.append(g_mtpa)
        g_fw = np.zeros(dim)
        g_fw[0] = -self.curr_max * 0.9
        g_fw[1] = self.curr_max * 0.1
        candidates.append(g_fw)
        return candidates


def ieee_machine2_params() -> PMSMParams:
    """Parameters for the 5-phase IEEE-Machine-2 prototype. R_stat, L_stat,
    flux_pm identified by ``utils.flux_fit.fit_pm_flux`` via joint least
    squares against measured voltage and torque in
    ``data/aggregated_file_means.csv``."""
    return PMSMParams(
        n_phases=5,
        n_ppairs=8,
        R_stat=np.diag([0.0191, 0.0514, 0.0805, 0.0801]),
        L_stat=1e-3
        * np.array(
            [
                [0.0920, -0.0286, -0.0141, 0.0010],
                [-0.0133, 0.1090, -0.0008, -0.0092],
                [-0.0088, 0.0037, 0.0725, -0.0466],
                [-0.0041, -0.0053, 0.0475, 0.0722],
            ]
        ),
        flux_pm=np.array([1.13438169e-02, 1.71999345e-03, 1.57771228e-05, 1.56271556e-05]),
    )


def ieee_machine2(
    *,
    curr_max: float = 30.0,
    volt_max: float = 13.0,
    omega_max: float = 1800.0,
) -> PMSMDrive:
    """5-phase IEEE-Machine-2 prototype. ``curr_max``/``volt_max``/``omega_max``
    are converter/controller limits, not machine physics, so they're
    overridable here rather than baked into ``PMSMParams``."""
    machine = PMSMDrive(ieee_machine2_params())
    machine.set_max_pars(curr_max, volt_max, omega_max)
    return machine


def neural_pmsm5phase(
    net: Any,
    scaler: Any,
    device: torch.device,
    *,
    curr_max: float = 30.0,
    volt_max: float = 13.0,
    omega_max: float = 1800.0,
) -> PMSMDrive:
    """IEEE-Machine-2 with a neural torque residual composed in."""
    machine = PMSMDrive(ieee_machine2_params(), torque_residual=NeuralTorqueResidual(net, scaler, device))
    machine.set_max_pars(curr_max, volt_max, omega_max)
    return machine


def neural_flux_pmsm5phase(
    net: Any,
    scaler: Any,
    device: torch.device,
    *,
    curr_max: float = 30.0,
    volt_max: float = 13.0,
    omega_max: float = 1800.0,
) -> PMSMDrive:
    """IEEE-Machine-2 with a neural flux model composed in place of the
    default ``ConstantFlux``."""
    params = ieee_machine2_params()
    cross_coupling = _build_cross_coupling(2)
    flux = NeuralFlux(net, scaler, device, params.L_stat, cross_coupling)
    machine = PMSMDrive(params, flux=flux)
    machine.set_max_pars(curr_max, volt_max, omega_max)
    return machine


# Trained flux-network artifact shipped in weights/ (see notebooks/flux_nn_trainer.ipynb
# "consistent-L training" section for how it was produced): a single-hidden-layer MLP,
# hidden_size=24, GELU, input (omega, i_d1, i_q1, i_d3, i_q3) -> output flux_pm (dim=4).
_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights"
_FLUX_NN_HIDDEN_SIZE = 24
_FLUX_NN_ACTIVATION = "gelu"
_FLUX_NN_INPUT_SIZE = 5
_FLUX_NN_OUTPUT_SIZE = 4


def ieee_machine2_trained_neural_flux(
    *,
    device: torch.device | None = None,
    curr_max: float = 30.0,
    volt_max: float = 13.0,
    omega_max: float = 1800.0,
) -> PMSMDrive:
    """IEEE-Machine-2 with the trained neural flux model from ``weights/``
    (``FluxNN_Weights.pth``/``FluxNN_Scaler.npy``) loaded and composed in.
    One-line equivalent of loading the net/scaler yourself and calling
    ``neural_flux_pmsm5phase(net, scaler, device)``."""
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
    return neural_flux_pmsm5phase(net, scaler, device, curr_max=curr_max, volt_max=volt_max, omega_max=omega_max)


@dataclass
class IMParams:
    """Data for one concrete 9-phase induction-motor prototype."""

    n_phases: int
    n_ppairs: int
    R_s: float
    R_r: np.ndarray
    L_mu: np.ndarray
    L_s_sigma: np.ndarray
    L_r_sigma: np.ndarray
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


def im9_prototype_params() -> IMParams:
    """Parameters for the 9-phase induction-motor prototype."""
    return IMParams(
        n_phases=9,
        n_ppairs=2,
        R_s=5.0,
        R_r=np.diag([1.54, 1.54, 1.57, 1.57]),
        L_mu=1e-3 * np.diag([496.0, 496.0, 58.2, 58.2]),
        L_s_sigma=1e-3 * np.diag([15.1, 15.1, 13.6, 13.6]),
        L_r_sigma=1e-3 * np.diag([53.3, 53.3, 33.4, 33.4]),
    )


def im9_prototype(
    *,
    curr_max: float = 20.0,
    volt_max: float = 200.0,
    omega_max: float = 1500.0,
) -> InductionDrive:
    machine = InductionDrive(im9_prototype_params())
    machine.set_max_pars(curr_max, volt_max, omega_max)
    return machine
