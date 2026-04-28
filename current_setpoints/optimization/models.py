from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch

from ..data import BaseMachine
from ..utils import NeuralTorquePredictor, predict_torque_neural


class BaseTorqueModel(ABC):
    """
    Abstract Base Class for motor torque models.
    Provides a unified interface for both analytical and neural-network-based torque calculations
    for any n-phase machine.
    """

    def __init__(self, machine: BaseMachine) -> None:
        """
        Initializes the torque model with machine parameters.

        Args:
            machine: Object containing motor constants and dynamic state logic.
        """
        self.machine = machine
        self.omega: float = 0.0

    def set_omega(self, omega: float) -> None:
        """
        Syncs the model with the current electrical speed of the optimization loop.

        Args:
            omega: Electrical speed in rad/s.
        """
        self.omega = omega

    @abstractmethod
    def calculate_torque(self, vec_curr_dq: np.ndarray) -> float:
        """
        Calculates torque for a given current vector.
        Must be implemented by subclasses.

        Args:
            vec_curr_dq: N-element current array (e.g., length 4 for 5-phase, 8 for 9-phase).

        Returns:
            float: Calculated electromagnetic torque in Nm.
        """
        pass

    def get_candidates(
        self, vec_curr_dq_guess: np.ndarray | None = None
    ) -> list[np.ndarray]:
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
            vec_curr_dq_guess: Optional user-provided warm-start vector.

        Returns:
            List[np.ndarray]: List of N-element current vectors to be used as starting points.
        """
        dim = self.machine.n_phases - 1
        candidates: list[np.ndarray] = []

        if vec_curr_dq_guess is not None:
            candidates.append(vec_curr_dq_guess.copy())
        else:
            default_guess = np.zeros(dim)
            default_guess[0] = 1.0
            candidates.append(default_guess)

        g_mtpa = np.zeros(dim)
        g_mtpa[1] = self.machine.curr_max * 0.95
        candidates.append(g_mtpa)

        g_fw = np.zeros(dim)
        g_fw[0] = -self.machine.curr_max * 0.9
        g_fw[1] = self.machine.curr_max * 0.1
        candidates.append(g_fw)

        return candidates


class ModelAnalytical(BaseTorqueModel):
    """
    Classic physics-based torque model.
    Uses the matrix form: T = i^T A i + 2 b^T i.
    Supports dynamic flux maps by updating the machine state before calculation.
    """

    def __init__(self, machine: BaseMachine) -> None:
        super().__init__(machine)

    def calculate_torque(self, vec_curr_dq: np.ndarray) -> float:
        """
        Computes torque using the machine's analytical quadratic form.
        Dynamically updates the machine's flux state before evaluation.
        """
        self.machine.update_state(self.omega, vec_curr_dq)

        return float(
            vec_curr_dq @ self.machine.mat_A @ vec_curr_dq
            + 2 * self.machine.vec_b @ vec_curr_dq
        )


class ModelNeural(BaseTorqueModel):
    """
    Neural-network-based torque model.
    Used for NTM-compensated calculations or complex saturation models where
    analytical equations are insufficient.
    """

    def __init__(
        self,
        machine: BaseMachine,
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
        super().__init__(machine)
        self.neural_model = neural_model
        self.scaler = scaler
        self.device = device

    def calculate_torque(self, vec_curr_dq: np.ndarray) -> float:
        """
        Computes torque by passing current and speed through the neural network.
        """
        return predict_torque_neural(
            vec_curr_dq=vec_curr_dq,
            omega=self.omega,
            neural_model=self.neural_model,
            scaler=self.scaler,
            device=self.device,
            machine=self.machine,
        )
