import numpy as np
import warnings
from typing import Optional, Union, List, Any


class MachineData:
    """
    Container class for motor grid data. Handles dimensionality validation,
    subsampling, and coordinate grid generation for motor performance mapping.
    """

    def __init__(
        self,
        torq: Union[np.ndarray, List[float]],
        omega: Union[np.ndarray, List[float]],
        segments: np.ndarray,
        isd1: np.ndarray,
        isd3: np.ndarray,
        isq1: np.ndarray,
        isq3: np.ndarray,
        k_skip: Optional[int] = None,
    ) -> None:
        """
        Initializes MachineData with torque/speed vectors and current component matrices.

        Args:
            torq: Torque vector or list.
            omega: Speed vector or list.
            segments: Matrix representing active constraint segments.
            isd1: Matrix of d1-axis currents.
            isd3: Matrix of d3-axis currents.
            isq1: Matrix of q1-axis currents.
            isq3: Matrix of q3-axis currents.
            k_skip: Optional factor to downsample the grid (e.g., take every k-th point).
        """
        self.torq: np.ndarray = np.array(torq).flatten()
        self.omega: np.ndarray = np.array(omega).flatten()
        self.segments: np.ndarray = np.array(segments)
        self.isd1: np.ndarray = np.array(isd1)
        self.isd3: np.ndarray = np.array(isd3)
        self.isq1: np.ndarray = np.array(isq1)
        self.isq3: np.ndarray = np.ndarray(isq3)

        # 1. Check Dimensions
        self._check_dimensions()

        # TODO: (DONE) remove all the Matlab indexing artefacts
        if k_skip is not None and k_skip > 1:
            self.select_k(k_skip)

        self.torq_grid: np.ndarray
        self.omega_grid: np.ndarray
        self.torq_grid, self.omega_grid = np.meshgrid(
            self.torq, self.omega, indexing="ij"
        )

        self.unique_segments: np.ndarray = np.unique(
            self.segments[~np.isnan(self.segments)]
        ).astype(int)

    def _check_dimensions(self) -> None:
        """
        Ensures all input arrays are compatible with torq and omega dimensions.

        Raises:
            ValueError: If any current or segment matrix shape does not match (len(torq), len(omega)).
        """
        n_torq = len(self.torq)
        n_omega = len(self.omega)

        expected_shape = (n_torq, n_omega)

        if self.segments.shape != expected_shape:
            # TODO: (DONE) never ever use this. use raise instead. this is horrible for tests, when it crashes Python instead of a desired error.
            raise ValueError(
                f"Segments shape {self.segments.shape} != expected {expected_shape}"
            )

        if self.isd1.shape != expected_shape:
            raise ValueError(
                "Current component dimensions do not match grid (torq x omega)."
            )

        # Sanity check on values (optional but recommended)
        # TODO: (DONE) should it be like this? Is negative speed ok? YES
        if np.any(self.omega < 0):
            warnings.warn(
                "Negative speeds detected in grid data. Ensure reverse rotation is intended.",
                UserWarning,
            )

    def select_k(self, k_skip: int) -> None:
        """
        Subsamples the torque/speed vectors and all associated matrices.

        Args:
            k_skip: The step size for slicing the arrays.
        """
        # TODO: (DONE) why is this here?
        self.omega = self.omega[::k_skip]
        self.torq = self.torq[::k_skip]

        self.segments = self.segments[::k_skip, ::k_skip]
        self.isd1 = self.isd1[::k_skip, ::k_skip]
        self.isd3 = self.isd3[::k_skip, ::k_skip]
        self.isq1 = self.isq1[::k_skip, ::k_skip]
        self.isq3 = self.isq3[::k_skip, ::k_skip]
