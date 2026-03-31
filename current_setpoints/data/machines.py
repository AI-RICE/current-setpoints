import numpy as np
from typing import Any, List, Union, Optional


class BaseMachine:
    """
    Abstract foundation for electrical machine models. Defines common physical 
    constants and structural validation logic for multiphase systems.
    """

    def __init__(self, n_phases: int, n_ppairs: int) -> None:
        """
        Initializes the base machine constants.

        Args:
            n_phases: Number of physical phases.
            n_ppairs: Number of pole pairs.
        """
        self.n_phases: int = n_phases          
        self.n_ppairs: int = n_ppairs        
        self.k_phase: float = n_phases / 2 
        self.mat_crossc: np.ndarray = np.array([ 
            [0, -1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, -3],
            [0, 0, 3, 0]
        ])

    def set_max_pars(self, curr_max: float, volt_max: float, omega_max: float) -> None:
        """
        Sets the global physical limits for the machine.

        Args:
            curr_max: Maximum peak phase current [A].
            volt_max: Maximum peak phase voltage [V].
            omega_max: Maximum electrical speed [rad/s].
        """
        self.curr_max: float = curr_max
        self.volt_max: float = volt_max
        self.omega_max: float = omega_max

    def check_data(self) -> None:
        """
        Validates that all internal matrices and vectors match the expected 
        dimensions based on the number of phases.
        """
        self._check_matrix(self.mat_A, False, 'mat_A')
        self._check_matrix(self.L_stat, False, 'L_stat')
        self._check_matrix(self.R_stat, True, 'R_stat')
        self._check_vector(self.vec_b, 'vec_b')

    def _check_matrix(self, mat: np.ndarray, allow_scalar: bool, name: str) -> None:
        """
        Internal helper to verify square matrix dimensions.

        Args:
            mat: Matrix to check.
            allow_scalar: If True, skips the n_phases-1 size check.
            name: Display name of the matrix for error messages.
        """
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]: 
            raise ValueError(f"Matrix {name} must be square.")
        if not allow_scalar and mat.shape[0] != self.n_phases - 1:
            raise ValueError(f"Matrix {name} must be square and sized to n_phases-1 ({self.n_phases - 1}).")

    def _check_vector(self, vec: np.ndarray, name: str) -> None:
        """
        Internal helper to verify vector dimensions.

        Args:
            vec: Vector to check.
            name: Display name of the vector for error messages.
        """
        if vec.ndim > 1 and vec.shape[1] != 1:
            raise ValueError(f"Vector {name} must be a column vector.")
        if vec.shape[0] != self.n_phases - 1:
            raise ValueError(f"Vector {name} must be a column vector with length n_phases-1 ({self.n_phases - 1}).")


# TODO: (DONE) normal names
class GenericMachine(BaseMachine):
    """
    Implementation of a machine model based on resistance, inductance, 
    and flux linkage vectors.
    """

    def __init__(
        self, 
        n_phases: int, 
        n_ppairs: int, 
        R_stat_vec: Union[np.ndarray, List[float]], 
        L_stat: np.ndarray, 
        flux_volt_vec: Union[np.ndarray, List[float]], 
        flux_torq_vec: Union[np.ndarray, List[float]]
    ) -> None:
        """
        Initializes the generic machine with specific electrical parameters.

        Args:
            n_phases: Number of physical phases.
            n_ppairs: Number of pole pairs.
            R_stat_vec: Stator resistance values for each harmonic subspace.
            L_stat: Stator inductance matrix.
            flux_volt_vec: Flux linkage vector for voltage calculation (Back-EMF).
            flux_torq_vec: Flux linkage vector used for torque calculation.
        """
        super().__init__(n_phases, n_ppairs)
        
        R_stat_vec = np.array(R_stat_vec).flatten() 
        flux_volt_vec = np.array(flux_volt_vec).flatten()
        flux_torq_vec = np.array(flux_torq_vec).flatten()
        L_stat = np.array(L_stat)

        self.flux_volt: np.ndarray = flux_volt_vec  
        self.flux_torq: np.ndarray = flux_torq_vec
        self.R_stat: np.ndarray = np.diag(R_stat_vec)
        self.L_stat: np.ndarray = L_stat
        self.mat_A: np.ndarray = np.zeros((4, 4))        
        
        self.vec_b: np.ndarray = (self.n_phases * self.n_ppairs / 4) * (self.mat_crossc @ self.flux_torq)                        
        self.check_data()


class IEEEMachine2(GenericMachine):
    """
    Specific 5-phase machine model based on standard IEEE benchmark data.
    """

    def __init__(self) -> None:
        """
        Initializes the IEEEMachine2 with hardcoded benchmark parameters.
        """
        n_phases = 5 
        n_ppairs = 8 
        R_stat_vec = [0.0191, 0.0514, 0.0805, 0.0801]         
        L_stat = np.array([ 
            [ 0.0920, -0.0286, -0.0141,  0.0010],
            [-0.0133,  0.1090, -0.0008, -0.0092],
            [-0.0088,  0.0037,  0.0725, -0.0466],
            [-0.0041, -0.0053,  0.0475,  0.0722]
        ]) * 1e-3      
        flux_volt_vec = [0.0115, 0.0018, 0, 0] 
        flux_torq_vec = [1.12810358e-02, -6.28421072e-04, 1.55053034e-04, -4.81016476e-05] 
        
        super().__init__(n_phases, n_ppairs, R_stat_vec, L_stat, flux_volt_vec, flux_torq_vec)