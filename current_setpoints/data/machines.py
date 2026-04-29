import numpy as np


class BaseMachine:
    """
    Abstract foundation for electrical machine models. Defines common physical
    constants, structural validation logic, and enforces implementation
    rules for multiphase systems.
    """

    def __init__(
        self,
        n_phases: int,
        n_ppairs: int,
        R_stat: np.ndarray,
        L_stat: np.ndarray,
        curr_max: float = 0.0,
        volt_max: float = 0.0,
        omega_max: float = 0.0,
    ) -> None:
        """
        Initializes the base machine constants and core matrices.

        Args:
            n_phases: Number of physical phases.
            n_ppairs: Number of pole pairs.
            R_stat: Stator resistance matrix.
            L_stat: Stator inductance matrix.
            curr_max: Maximum peak phase current [A].
            volt_max: Maximum peak phase voltage [V].
            omega_max: Maximum mechanical speed [RPM]. Consumed by grid.py
                as ``omega_max / const_mech_speed`` to yield electrical rad/s.
        """
        self.n_phases: int = n_phases
        self.n_ppairs: int = n_ppairs
        self.k_phase: float = n_phases / 2

        self.R_stat: np.ndarray = R_stat
        self.L_stat: np.ndarray = L_stat
        self.curr_max: float = curr_max
        self.volt_max: float = volt_max
        self.omega_max: float = omega_max
        self.dim = self.n_phases - 1

        self.mat_crossc: np.ndarray = np.zeros((self.dim, self.dim))
        for i in range(self.dim // 2):
            h = 2 * i + 1
            self.mat_crossc[2 * i, 2 * i + 1] = -h
            self.mat_crossc[2 * i + 1, 2 * i] = h

    # def qwe(self):
    #     self.mat_A: np.ndarray = np.zeros((dim, dim))
    #     self.vec_b: np.ndarray = np.zeros(dim)
    #     self.flux_volt: np.ndarray = np.zeros(dim)
    #     self.flux_torq: np.ndarray = np.zeros(dim)



    # @abstractmethod
    # def update_state(self, omega: float, vec_curr_dq: np.ndarray) -> None:
    #     """
    #     Abstract method. Forces any child class to implement their own
    #     logic for updating internal flux vectors and dependent states.
    #     """
    #     pass

    def set_max_pars(self, curr_max: float, volt_max: float, omega_max: float) -> None:
        """
        Sets the global physical limits for the machine.

        Args:
            curr_max: Maximum peak phase current [A].
            volt_max: Maximum peak phase voltage [V].
            omega_max: Maximum mechanical speed [RPM].
        """
        self.curr_max = curr_max
        self.volt_max = volt_max
        self.omega_max = omega_max

    def check_data(self) -> None:
        """
        Validates that all internal matrices and vectors match the expected
        dimensions based on the number of phases.
        """
        # self._check_matrix(self.mat_A, "mat_A")
        self._check_matrix(self.L_stat, "L_stat")
        self._check_matrix(self.R_stat, "R_stat")
        # self._check_vector(self.vec_b, "vec_b")

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
        if vec.ndim == 0:
            raise ValueError(f"Vector {name} must be 1D or 2D, not a 0-D scalar.")
        if vec.ndim > 1 and vec.shape[1] != 1:
            raise ValueError(f"Vector {name} must be a 1D or column vector.")
        if vec.shape[0] != self.n_phases - 1:
            raise ValueError(
                f"Vector {name} must have length n_phases-1 ({self.n_phases - 1}). "
                f"Got length {vec.shape[0]}."
            )

class IEEEMachine2(BaseMachine):
    """
    Specific 5-phase machine model based on standard IEEE benchmark data.
    """

    def __init__(self) -> None:
        """
        Initializes the IEEEMachine2 with benchmark electrical parameters and
        connects it to the centralized flux provider. Flux constants for this
        machine are registered by FluxValues itself (see parameters.py).
        """
        n_phases = 5
        n_ppairs = 8
        # np.diag(np.array(R_stat_vec).flatten())
        # TODO: check
        R_stat = np.diag([0.0191, 0.0514, 0.0805, 0.0801])
        L_stat = 1e-3 * np.array([
                    [0.0920, -0.0286, -0.0141, 0.0010],
                    [-0.0133, 0.1090, -0.0008, -0.0092],
                    [-0.0088, 0.0037, 0.0725, -0.0466],
                    [-0.0041, -0.0053, 0.0475, 0.0722],
                ])

        super().__init__(
            n_phases=n_phases,
            n_ppairs=n_ppairs,
            R_stat=R_stat,
            L_stat=L_stat,
        )
