from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch

from ..parameters import BaseMachine, Flux
from ..utils import (
    NeuralTorquePredictor,
    predict_torque_neural,
    substitution2_loss_features,
    substitution_loss_features,
)


class BaseTorqueModel(ABC):
    """
    Abstract Base Class for motor torque models.
    Provides a unified interface for both analytical and neural-network-based torque calculations
    for any n-phase machine.
    """

    def __init__(self, curr_max: float, n_phases: int) -> None:
        """
        Initializes the torque model with machine parameters.

        Args:
            machine: Object containing motor constants and dynamic state logic.
        """
        self.curr_max = curr_max
        self.n_phases = n_phases

    @abstractmethod
    def calculate_torque(self, omega: float, curr_dq: np.ndarray) -> float:
        """
        Calculates torque for a given current vector.
        Must be implemented by subclasses.

        Args:
            curr_dq: N-element current array (e.g., length 4 for 5-phase, 8 for 9-phase).

        Returns:
            float: Calculated electromagnetic torque in Nm.
        """
        pass

    def get_candidates(self, curr_dq_guess: np.ndarray | None = None) -> list[np.ndarray]:
        """
        Generates a list of initial guess vectors for the optimizer.
        These candidates help the solver avoid local minima by providing starts in
        MTPA and Flux-Weakening regions.

        Note: the MTPA and FW seed vectors only populate indices [0] and [1]
        (the fundamental DQ subspace), leaving higher-harmonic components at
        zero. This is well-suited to 5-phase machines where indices 2-3 are
        the 3rd-harmonic subspace and starting from "no 3rd-harmonic
        injection" is reasonable; richer warm starts may be desirable for
        7-phase or 9-phase machines.

        Args:
            curr_dq_guess: Optional user-provided warm-start vector.

        Returns:
            List[np.ndarray]: List of N-element current vectors to be used as starting points.
        """
        dim = self.n_phases - 1
        candidates: list[np.ndarray] = []

        # Always include the default seed. A warm start (when provided) AUGMENTS
        # the candidate set rather than replacing the default, because different
        # operating points converge from different seeds: the field-weakening
        # corner needs the warm start, while some points only converge from the
        # default. Replacing the default could drop the only seed that works at a
        # given speed (leaving maximize_torque non-convergent there).
        default_guess = np.zeros(dim)
        default_guess[0] = 1.0
        candidates.append(default_guess)

        if curr_dq_guess is not None:
            candidates.append(curr_dq_guess.copy())

        g_mtpa = np.zeros(dim)
        g_mtpa[1] = self.curr_max * 0.95
        candidates.append(g_mtpa)

        g_fw = np.zeros(dim)
        g_fw[0] = -self.curr_max * 0.9
        g_fw[1] = self.curr_max * 0.1
        candidates.append(g_fw)

        return candidates


class ModelAnalytical(BaseTorqueModel):
    """
    Classic physics-based torque model.
    Uses the matrix form: T = i^T A i + 2 b^T i.
    Supports dynamic flux maps by updating the machine state before calculation.
    """

    def __init__(self, machine: BaseMachine, flux: Flux) -> None:
        self.A = (machine.n_phases * machine.n_ppairs / 4.0) * (
            machine.mat_crossc @ machine.L_stat + machine.L_stat @ machine.mat_crossc.T
        )
        self.machine = machine
        self.n_ppairs = machine.n_ppairs
        self.flux = flux
        self.mat_crossc = machine.mat_crossc
        super().__init__(machine.curr_max, machine.n_phases)

    def calculate_torque(self, omega: float, curr_dq: np.ndarray) -> float:
        """
        Computes torque using the machine's analytical quadratic form.
        Dynamically updates the machine's flux state before evaluation.
        """

        _, flux_torq = self.flux.get_flux(omega, curr_dq)
        b = self.n_phases * self.n_ppairs / 4 * (self.mat_crossc @ flux_torq)

        return float(curr_dq @ self.A @ curr_dq + 2 * b @ curr_dq)


class ModelNeural(ModelAnalytical):
    """
    Neural-network-based torque model.
    Used for NTM-compensated calculations or complex saturation models where
    analytical equations are insufficient.
    """

    def __init__(
        self,
        machine: BaseMachine,
        flux: Flux,
        neural_model: NeuralTorquePredictor,
        scaler: Any,
        device: torch.device,
    ) -> None:
        """
        Initializes the neural model with the trained network and scaling logic.

        Args:
            machine: Machine parameter object.
            neural_model: Trained PyTorch model implementing the (X, B) -> torque interface.
            scaler: Input scaler (e.g., sklearn StandardScaler) implementing
                ``transform``. Typed as Any because sklearn scalers do not share
                a clean abstract base class.
            device: Torch computation device (e.g., torch.device('cpu') or 'cuda').
        """
        super().__init__(machine, flux)
        self.neural_model = neural_model
        self.scaler = scaler
        self.device = device

    def calculate_torque(self, omega: float, curr_dq: np.ndarray) -> float:
        """
        Computes torque by passing current and speed through the neural network.
        """

        torque_analytical = super().calculate_torque(omega, curr_dq)
        torque_residual = predict_torque_neural(
            curr_dq=curr_dq, omega=omega, neural_model=self.neural_model, scaler=self.scaler, device=self.device
        )

        return torque_analytical + torque_residual


class ModelLossesSubstitution1:
    """
    Analytical torque plus a simple voltage-substitution iron-loss correction.

    The total torque is ``T_base + M_Fe``, where ``T_base`` is the analytical
    baseline and ``M_Fe`` is an iron-loss term built from the dq voltages
    (fundamental and 3rd-harmonic subspaces):

        M_Fe = k_v * (p^2 * omega_m / (4 pi^2)) * (U1^2 + 9 U3^2)   # eddy-type
             + k_h * (p / (2 pi))              * (U1^2 + 3 U3^2)    # hysteresis-type

    with ``U1^2 = U_d1^2 + U_q1^2``, ``U3^2 = U_d3^2 + U_q3^2``,
    ``omega_m = omega / p`` the mechanical speed, and ``p`` the pole-pair count.

    The coefficients ``k_v, k_h`` are identified by least squares against the
    measured residual; see ``current_setpoints.utils.loss_fit``. This model is
    intended for residual benchmarking against the neural torque model, not for
    optimizer-driven map generation. It deliberately does NOT subclass
    ``ModelAnalytical``: its ``calculate_torque`` needs the dq voltage vector
    explicitly (the measured voltages from the dataset), an incompatible
    signature with the optimizer-facing models, so it composes a baseline model
    internally instead of inheriting one.
    """

    def __init__(self, machine: BaseMachine, flux: Flux, k_v: float = 0.0, k_h: float = 0.0) -> None:
        """
        Args:
            machine: Machine parameter object (supplies ``n_ppairs`` = ``p``).
            flux: Flux provider used by the analytical baseline.
            k_v: Eddy-type loss coefficient.
            k_h: Hysteresis-type loss coefficient.
        """
        self.baseline = ModelAnalytical(machine, flux)
        self.n_ppairs = machine.n_ppairs
        self.k_v = k_v
        self.k_h = k_h

    def iron_loss(self, omega: float, volt_dq: np.ndarray) -> float:
        """
        Iron-loss torque correction ``M_Fe`` at one operating point.

        Args:
            omega: Electrical speed [rad/s].
            volt_dq: dq voltage vector ``[U_d1, U_q1, U_d3, U_q3]``.

        Returns:
            float: ``M_Fe`` [Nm].
        """
        f_v, f_h = substitution_loss_features(omega, volt_dq, self.n_ppairs)
        return float(self.k_v * f_v + self.k_h * f_h)

    def calculate_torque(self, omega: float, curr_dq: np.ndarray, volt_dq: np.ndarray) -> float:
        """
        Total torque ``T_base + M_Fe``.

        Args:
            omega: Electrical speed [rad/s].
            curr_dq: dq current vector ``[i_d1, i_q1, i_d3, i_q3]`` (baseline).
            volt_dq: dq voltage vector ``[U_d1, U_q1, U_d3, U_q3]`` (loss term).

        Returns:
            float: Total torque [Nm].
        """
        return self.baseline.calculate_torque(omega, curr_dq) + self.iron_loss(omega, volt_dq)


class ModelLossSubstitution2:
    """
    Analytical torque plus an alternative voltage-substitution iron-loss term.

    Same structure as ``ModelLossesSubstitution1`` (total torque ``T_base +
    M_Fe``, measured dq voltages passed explicitly, composes a baseline model
    rather than subclassing it), but with a different substitution for ``M_Fe``:

        M_Fe = k_v / omega_m              * (U1^2 + U3^2)
             + k_h * 2 pi / (p * omega_m^2) * (U1^2 + (1/3) U3^2)

    with ``omega_m = omega / p``. Both terms diverge at standstill, so this
    model is only valid away from ``omega = 0``. Coefficients are identified by
    least squares; see ``current_setpoints.utils.loss_fit``.
    """

    def __init__(self, machine: BaseMachine, flux: Flux, k_v: float = 0.0, k_h: float = 0.0) -> None:
        """
        Args:
            machine: Machine parameter object (supplies ``n_ppairs`` = ``p``).
            flux: Flux provider used by the analytical baseline.
            k_v: Eddy-type loss coefficient.
            k_h: Hysteresis-type loss coefficient.
        """
        self.baseline = ModelAnalytical(machine, flux)
        self.n_ppairs = machine.n_ppairs
        self.k_v = k_v
        self.k_h = k_h

    def iron_loss(self, omega: float, volt_dq: np.ndarray) -> float:
        """
        Iron-loss torque correction ``M_Fe`` at one operating point.

        Args:
            omega: Electrical speed [rad/s].
            volt_dq: dq voltage vector ``[U_d1, U_q1, U_d3, U_q3]``.

        Returns:
            float: ``M_Fe`` [Nm].
        """
        f_v, f_h = substitution2_loss_features(omega, volt_dq, self.n_ppairs)
        return float(self.k_v * f_v + self.k_h * f_h)

    def calculate_torque(self, omega: float, curr_dq: np.ndarray, volt_dq: np.ndarray) -> float:
        """
        Total torque ``T_base + M_Fe``.

        Args:
            omega: Electrical speed [rad/s].
            curr_dq: dq current vector ``[i_d1, i_q1, i_d3, i_q3]`` (baseline).
            volt_dq: dq voltage vector ``[U_d1, U_q1, U_d3, U_q3]`` (loss term).

        Returns:
            float: Total torque [Nm].
        """
        return self.baseline.calculate_torque(omega, curr_dq) + self.iron_loss(omega, volt_dq)
