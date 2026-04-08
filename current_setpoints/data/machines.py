import numpy as np
from typing import Union, List
from .parameters import FluxValues


class BaseMachine:
    """
    Abstract foundation for electrical machine models. Defines common physical
    constants and structural validation logic for multiphase systems.
    """

    mat_A: np.ndarray
    L_stat: np.ndarray
    R_stat: np.ndarray
    vec_b: np.ndarray
    curr_max: float
    volt_max: float
    omega_max: float

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

        # Dynamically generate the cross-coupling matrix based on phase count.
        # This maps the harmonic subspaces (1, 3, 5, 7...) to block diagonals.
        dim = self.n_phases - 1
        self.mat_crossc: np.ndarray = np.zeros((dim, dim))
        for i in range(dim // 2):
            h = 2 * i + 1  # Harmonic number
            self.mat_crossc[2 * i, 2 * i + 1] = -h
            self.mat_crossc[2 * i + 1, 2 * i] = h

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
        # Strictly enforce dimensions on all structural matrices
        self._check_matrix(self.mat_A, "mat_A")
        self._check_matrix(self.L_stat, "L_stat")
        self._check_matrix(self.R_stat, "R_stat")
        self._check_vector(self.vec_b, "vec_b")

    def _check_matrix(self, mat: np.ndarray, name: str) -> None:
        """
        Internal helper to strictly verify square matrix dimensions.
        """
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            raise ValueError(f"Matrix {name} must be square.")
        if mat.shape[0] != self.n_phases - 1:
            raise ValueError(
                f"Matrix {name} must be sized to n_phases-1 ({self.n_phases - 1}x{self.n_phases - 1}). "
                f"Got {mat.shape[0]}x{mat.shape[1]}."
            )

    def _check_vector(self, vec: np.ndarray, name: str) -> None:
        """
        Internal helper to strictly verify vector dimensions.
        """
        if vec.ndim > 1 and vec.shape[1] != 1:
            raise ValueError(f"Vector {name} must be a 1D or column vector.")
        if vec.shape[0] != self.n_phases - 1:
            raise ValueError(
                f"Vector {name} must have length n_phases-1 ({self.n_phases - 1}). "
                f"Got length {vec.shape[0]}."
            )


class GenericMachine(BaseMachine):
    """
    Implementation of a machine model based on resistance, inductance,
    and a dynamic/constant flux provider.
    """

    def __init__(
        self,
        n_phases: int,
        n_ppairs: int,
        R_stat_vec: Union[np.ndarray, List[float]],
        L_stat: np.ndarray,
        machine_name: str,
        flux_values: FluxValues,
    ) -> None:
        """
        Initializes the generic machine with specific electrical parameters and a flux provider.

        Args:
            n_phases: Number of physical phases.
            n_ppairs: Number of pole pairs.
            R_stat_vec: Stator resistance values for each harmonic subspace.
            L_stat: Stator inductance matrix.
            machine_name: String identifier for the flux provider to look up data.
            flux_values: Dependency-injected provider for fetching flux.
        """
        super().__init__(n_phases, n_ppairs)

        self.machine_name = machine_name
        self.flux_values = flux_values

        R_stat_vec = np.array(R_stat_vec).flatten()
        self.R_stat: np.ndarray = np.diag(R_stat_vec)
        self.L_stat: np.ndarray = np.array(L_stat)

        # 1. Initialize the required quadratic parameter matrix
        dim = self.n_phases - 1
        self.mat_A: np.ndarray = np.zeros((dim, dim))

        # 2. Bootstrap the dynamic machine state at 0 speed and 0 current
        dummy_vec_curr_dq = np.zeros(dim)
        self.update_state(omega=0.0, vec_curr_dq=dummy_vec_curr_dq)

        # 3. Validate matrix/vector dimensions
        self.check_data()

    def update_state(self, omega: float, vec_curr_dq: np.ndarray) -> None:
        """
        Fetches updated flux values from the provider based on the current operating
        point and recalculates the dynamic dependent vector (vec_b).

        This MUST be called by your simulation/optimization loop whenever currents
        or speed change.
        """
        self.flux_volt, self.flux_torq = self.flux_values.get_flux(
            self.machine_name, omega, vec_curr_dq
        )

        # Runtime validation: ensure fetched fluxes match our phase constraints
        self._check_vector(self.flux_torq, "flux_torq (from FluxValues)")
        self._check_vector(self.flux_volt, "flux_volt (from FluxValues)")

        # vec_b relies on flux_torq, so it updates whenever the operating point changes
        self.vec_b = (self.n_phases * self.n_ppairs / 4) * (
            self.mat_crossc @ self.flux_torq
        )


class IEEEMachine2(GenericMachine):
    """
    Specific 5-phase machine model based on standard IEEE benchmark data.
    """

    def __init__(self, flux_values: FluxValues) -> None:
        """
        Initializes the IEEEMachine2 with benchmark electrical parameters and connects
        it to the centralized flux provider.
        """
        n_phases = 5
        n_ppairs = 8
        R_stat_vec = [0.0191, 0.0514, 0.0805, 0.0801]
        L_stat = (
            np.array(
                [
                    [0.0920, -0.0286, -0.0141, 0.0010],
                    [-0.0133, 0.1090, -0.0008, -0.0092],
                    [-0.0088, 0.0037, 0.0725, -0.0466],
                    [-0.0041, -0.0053, 0.0475, 0.0722],
                ]
            )
            * 1e-3
        )

        super().__init__(
            n_phases=n_phases,
            n_ppairs=n_ppairs,
            R_stat_vec=R_stat_vec,
            L_stat=L_stat,
            machine_name="IEEEMachine2",
            flux_values=flux_values,
        )
