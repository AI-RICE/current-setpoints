import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Any
from utils.neural import predict_torque_neural


class BaseTorqueModel(ABC):
    """
    Abstract Base Class for motor torque models.
    Provides a unified interface for both analytical and neural-network-based torque calculations.
    """

    def __init__(self, machine: Any) -> None:
        """
        Initializes the torque model with machine parameters.

        Args:
            machine: Object containing motor constants (e.g., A matrix, b vector, curr_max).
        """
        self.machine = machine

    @abstractmethod
    def calculate_torque(self, vec_curr_dq: np.ndarray) -> float:
        """
        Calculates torque for a given current vector.
        Must be implemented by subclasses.

        Args:
            vec_curr_dq: 4-element current array [id1, iq1, id3, iq3].

        Returns:
            float: Calculated electromagnetic torque in Nm.
        """
        pass

    def get_candidates(
        self, vec_curr_dq_guess: Optional[np.ndarray] = None
    ) -> List[np.ndarray]:
        """
        Generates a list of initial guess vectors for the optimizer.
        These candidates help the solver avoid local minima by providing starts in
        MTPA and Flux-Weakening regions.

        Args:
            vec_curr_dq_guess: Optional user-provided warm-start vector.

        Returns:
            List[np.ndarray]: List of 4-element current vectors to be used as starting points.
        """
        candidates: List[np.ndarray] = []

        # 1. Primary Guess: Use provided warm-start or a default small vector
        if vec_curr_dq_guess is not None:
            candidates.append(vec_curr_dq_guess)
        else:
            candidates.append(np.array([1.0, 0.0, 0.0, 0.0]))

        # 2. MTPA Guess: High Q-axis (Typical for maximum torque per ampere)
        g_mtpa = np.zeros(4)
        g_mtpa[1] = self.machine.curr_max * 0.95
        candidates.append(g_mtpa)

        # 3. Flux Weakening Guess: High Negative D-axis (Required for high-speed operation)
        g_fw = np.zeros(4)
        g_fw[0] = -self.machine.curr_max * 0.9
        g_fw[1] = self.machine.curr_max * 0.1
        candidates.append(g_fw)

        return candidates


class ModelAnalytical(BaseTorqueModel):
    """
    Classic physics-based torque model.
    Uses the matrix form: $T = \mathbf{i}^T \mathbf{A} \mathbf{i} + 2\mathbf{b} \mathbf{i}$.
    """

    def __init__(self, machine: Any) -> None:
        super().__init__(machine)

    def calculate_torque(self, vec_curr_dq: np.ndarray) -> float:
        """
        Computes torque using the machine's analytical quadratic form.
        """
        return (
            vec_curr_dq @ self.machine.A @ vec_curr_dq
            + 2 * self.machine.b @ vec_curr_dq
        )


class ModelNeural(BaseTorqueModel):
    """
    Neural-network-based torque model.
    Used for PIRN-compensated calculations or complex saturation models where
    analytical equations are insufficient.
    """

    def __init__(
        self, machine: Any, neural_model: Any, scaler: Any, device: Any, omega: float
    ) -> None:
        """
        Initializes the neural model with the trained network and scaling logic.

        Args:
            machine: Machine parameter object.
            neural_model: Loaded PyTorch/TensorFlow model.
            scaler: Input/Output scaler (e.g., StandardScaler) for normalization.
            device: Computation device (e.g., 'cpu' or 'cuda').
            omega: Electrical speed in rad/s.
        """
        super().__init__(machine)
        self.neural_model = neural_model
        self.scaler = scaler
        self.device = device
        self.omega = omega

    def calculate_torque(self, vec_curr_dq: np.ndarray) -> float:
        """
        Computes torque by passing current and speed through the neural network.
        """
        # Passing the instance variables into the utility function
        return predict_torque_neural(
            vec_curr_dq=vec_curr_dq,
            omega=self.omega,
            neural_model=self.neural_model,
            scaler=self.scaler,
            device=self.device,
        )
