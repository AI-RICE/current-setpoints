import numpy as np


class BaseMachine:
    """
    Abstract foundation for electrical machine models. Defines common physical
    constants, structural validation logic, and enforces implementation
    rules for multiphase systems.

    Subclasses must assign the following machine-identity attributes to
    ``self`` BEFORE calling ``super().__init__()``:
        n_phases, n_ppairs, R_stat, L_stat

    The operating limits (curr_max, volt_max, omega_max) are declared at
    class level so type checkers see them, but they are NOT initialised here.
    The user must call ``set_max_pars`` before any consumer reads them;
    otherwise an ``AttributeError`` is raised at access time.
    """

    n_phases: int
    n_ppairs: int
    R_stat: np.ndarray
    L_stat: np.ndarray

    curr_max: float
    volt_max: float
    omega_max: float

    def __init__(self) -> None:
        """
        Initializes derived constants and core matrices from attributes
        already set on ``self`` by the subclass.
        """
        self.k_phase: float = self.n_phases / 2
        self.dim: int = self.n_phases - 1

        self.mat_crossc: np.ndarray = np.zeros((self.dim, self.dim))
        for i in range(self.dim // 2):
            h = 2 * i + 1
            self.mat_crossc[2 * i, 2 * i + 1] = -h
            self.mat_crossc[2 * i + 1, 2 * i] = h

        self.check_data()

    def set_max_pars(self, curr_max: float, volt_max: float, omega_max: float) -> None:
        """
        Sets the global physical limits for the machine. Must be called
        before any consumer reads ``curr_max``, ``volt_max``, or ``omega_max``.

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
        self._check_matrix(self.L_stat, "L_stat")
        self._check_matrix(self.R_stat, "R_stat")

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


class IEEEMachine2(BaseMachine):
    """
    Specific 5-phase machine model based on standard IEEE benchmark data.
    """

    def __init__(self) -> None:
        self.n_phases: int = 5
        self.n_ppairs: int = 8

        self.R_stat: np.ndarray = np.diag([0.0191, 0.0514, 0.0805, 0.0801])
        self.L_stat: np.ndarray = 1e-3 * np.array([
                [0.0920, -0.0286, -0.0141, 0.0010],
                [-0.0133, 0.1090, -0.0008, -0.0092],
                [-0.0088, 0.0037, 0.0725, -0.0466],
                [-0.0041, -0.0053, 0.0475, 0.0722],
            ])

        super().__init__()
