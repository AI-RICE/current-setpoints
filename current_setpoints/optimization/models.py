import numpy as np
from abc import ABC, abstractmethod
from utils.neural import predict_torque_pirn

class BaseTorqueModel(ABC):
    def __init__(self, IPM):
        self.IPM = IPM
        
    @abstractmethod
    def calculate_torque(self, is_vec):
        """Calculates torque for a given current vector."""
        pass

    def get_candidates(self, is0=None):
        """Generates a list of initial guess vectors for the optimizer.
        Shared across all inheriting torque models.
        """
        candidates = []
        if is0 is not None:
            candidates.append(is0)
        else:
            candidates.append(np.array([1.0, 0.0, 0.0, 0.0]))
            
        # MTPA Guess (High Q-axis)
        g_mtpa = np.zeros(4)
        g_mtpa[1] = self.IPM.Imax * 0.95
        candidates.append(g_mtpa)
        
        # Flux Weakening Guess (High Negative D-axis)
        g_fw = np.zeros(4)
        g_fw[0] = -self.IPM.Imax * 0.9
        g_fw[1] = self.IPM.Imax * 0.1
        candidates.append(g_fw)
        
        return candidates


class ModelAnalytical(BaseTorqueModel):
    def __init__(self, IPM):
        super().__init__(IPM)

    def calculate_torque(self, is_vec):
        # Analytical Torque Equation
        return is_vec @ self.IPM.A @ is_vec + 2 * self.IPM.b @ is_vec


class ModelNeural(BaseTorqueModel):
    def __init__(self, IPM, pirn_model, scaler, device, omega):
        super().__init__(IPM)
        self.pirn_model = pirn_model
        self.scaler = scaler
        self.device = device
        self.omega = omega

    def calculate_torque(self, is_vec):
        # Passing the instance variables into the utility function
        return predict_torque_pirn(
            is_vec=is_vec, 
            omega=self.omega, 
            pirn_model=self.pirn_model, 
            scaler=self.scaler, 
            device=self.device
        )