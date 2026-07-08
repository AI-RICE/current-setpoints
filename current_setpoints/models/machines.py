from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch


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
        self._scaler_mean = torch.from_numpy(scaler.mean_).float().to(device)
        self._scaler_scale = torch.from_numpy(scaler.scale_).float().to(device)

    def flux(self, omega: float, curr_dq: np.ndarray) -> np.ndarray:
        X = self.scaler.transform(np.hstack(([omega], curr_dq)).reshape(1, -1))
        with torch.no_grad():
            out = self.net(torch.from_numpy(X).float().to(self.device))
        return out.cpu().numpy().reshape(-1)

    def inductance(self, omega: float, curr_dq: np.ndarray | None = None) -> np.ndarray:
        if curr_dq is None:
            curr_dq = np.zeros(self.L_stat.shape[0])
        x_raw = torch.tensor(np.hstack(([omega], curr_dq)), dtype=torch.float32, device=self.device)

        def f(x: torch.Tensor) -> torch.Tensor:
            x_normed = ((x - self._scaler_mean) / self._scaler_scale).unsqueeze(0)
            return self.net(x_normed).squeeze(0)

        jac = torch.autograd.functional.jacobian(f, x_raw)  # (dim, 1 + dim): d(flux_j)/d(x_k)
        d_flux_d_i = jac[:, 1:].detach().cpu().numpy()
        return self.L_stat + d_flux_d_i


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
        # Use the legacy global RNG with state save/restore to avoid numpy
        # RNG-constructor ABC recursion in older numpy versions.
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


class PMSM5Phase(DriveModel):
    def __init__(self) -> None:
        self.n_phases = 5
        self.n_harmonics = 2
        self.dim = 4
        self.n_ppairs = 8
        self.k_phase = 5 / 2

        self.R_stat = np.diag([0.0191, 0.0514, 0.0805, 0.0801])
        self.k_v = 0.0
        self.k_h = 0.0

        cross_coupling = _build_cross_coupling(self.n_harmonics)
        L_stat = 1e-3 * np.array(
            [
                [0.0920, -0.0286, -0.0141, 0.0010],
                [-0.0133, 0.1090, -0.0008, -0.0092],
                [-0.0088, 0.0037, 0.0725, -0.0466],
                [-0.0041, -0.0053, 0.0475, 0.0722],
            ]
        )
        # Identified by utils.flux_fit.fit_pm_flux via joint least squares
        # against measured voltage and torque in data/aggregated_file_means.csv.
        flux_pm = np.array([1.13438169e-02, 1.71999345e-03, 1.57771228e-05, 1.56271556e-05])

        self.flux: FluxModel = ConstantFlux(flux_pm, L_stat, cross_coupling)

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
        return float(curr_dq @ A @ curr_dq + 2 * b @ curr_dq)

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


class IM9Phase(DriveModel):
    def __init__(self) -> None:
        self.n_phases = 9
        self.n_harmonics = 2
        self.dim = 4
        self.n_ppairs = 2
        self.k_phase = 9 / 2

        R_s = 5.0
        self.R_stat = R_s * np.eye(self.dim)
        self.k_v = 0.0
        self.k_h = 0.0

        self._R_r = np.diag([1.54, 1.54, 1.57, 1.57])
        self._L_mu = 1e-3 * np.diag([496.0, 496.0, 58.2, 58.2])
        L_s_sigma = 1e-3 * np.diag([15.1, 15.1, 13.6, 13.6])
        L_r_sigma = 1e-3 * np.diag([53.3, 53.3, 33.4, 33.4])
        self._L_s = self._L_mu + L_s_sigma
        self._L_r = self._L_mu + L_r_sigma
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


class NeuralPMSM5Phase(PMSM5Phase):
    def __init__(
        self,
        net: Any,  # NeuralTorquePredictor
        scaler: Any,  # sklearn StandardScaler
        device: torch.device,
    ) -> None:
        super().__init__()
        self._net = net
        self._scaler = scaler
        self._device = device

    def torque(self, omega: float, curr_dq: np.ndarray) -> float:
        return super().torque(omega, curr_dq) + self._predict(omega, curr_dq)

    def _predict(self, omega: float, curr_dq: np.ndarray) -> float:
        X = self._scaler.transform(np.hstack(([omega], curr_dq)).reshape(1, -1))
        with torch.no_grad():
            residual = self._net(torch.from_numpy(X).float().to(self._device))
        return float(residual.cpu().numpy().item())


class NeuralFluxPMSM5Phase(PMSM5Phase):
    def __init__(
        self,
        net: Any,  # NeuralFluxPredictor
        scaler: Any,  # sklearn StandardScaler
        device: torch.device,
    ) -> None:
        super().__init__()
        self.flux: FluxModel = NeuralFlux(net, scaler, device, self.flux.L_stat, self.flux.cross_coupling)
