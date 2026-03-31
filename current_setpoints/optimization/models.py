import numpy as np
from abc import ABC, abstractmethod
from utils.neural import predict_torque_neural

class BaseTorqueModel(ABC):
    def __init__(self, machine):
        self.machine = machine
        
    @abstractmethod
    def calculate_torque(self, vec_curr_dq):
        """Calculates torque for a given current vector."""
        pass

    def get_candidates(self, vec_curr_dq_guess=None):
        """Generates a list of initial guess vectors for the optimizer.
        Shared across all inheriting torque models.
        """
        candidates = []
        if vec_curr_dq_guess is not None:
            candidates.append(vec_curr_dq_guess)
        else:
            candidates.append(np.array([1.0, 0.0, 0.0, 0.0]))
            
        # MTPA Guess (High Q-axis)
        g_mtpa = np.zeros(4)
        g_mtpa[1] = self.machine.curr_max * 0.95
        candidates.append(g_mtpa)
        
        # Flux Weakening Guess (High Negative D-axis)
        g_fw = np.zeros(4)
        g_fw[0] = -self.machine.curr_max * 0.9
        g_fw[1] = self.machine.curr_max * 0.1
        candidates.append(g_fw)
        
        return candidates


class ModelAnalytical(BaseTorqueModel):
    def __init__(self, machine):
        super().__init__(machine)

    def calculate_torque(self, vec_curr_dq):
        # Analytical Torque Equation
        return vec_curr_dq @ self.machine.A @ vec_curr_dq + 2 * self.machine.b @ vec_curr_dq


class ModelNeural(BaseTorqueModel):
    def __init__(self, machine, neural_model, scaler, device, omega):
        super().__init__(machine)
        self.neural_model = neural_model
        self.scaler = scaler
        self.device = device
        self.omega = omega

    def calculate_torque(self, vec_curr_dq):
        # Passing the instance variables into the utility function
        return predict_torque_neural(
            vec_curr_dq=vec_curr_dq, 
            omega=self.omega, 
            neural_model=self.neural_model, 
            scaler=self.scaler, 
            device=self.device
        )