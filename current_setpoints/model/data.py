import warnings
from typing import List, Optional, Union

import numpy as np


class MachineData:
    """
    Container class for motor grid data. Handles dimensionality validation,
    subsampling, and coordinate grid generation for motor performance mapping
    across any number of phases/harmonics.
    """

    def __init__(
        self,
        torq: Union[np.ndarray, List[float]],
        omega: Union[np.ndarray, List[float]],
        segments: np.ndarray,
        curr_dq_grid: np.ndarray,
        k_skip: Optional[int] = None,
    ) -> None:
        """
        Initializes MachineData with torque/speed vectors and current component matrices.

        Args:
            torq: Torque vector or list.
            omega: Speed vector or list.
            segments: 2D Matrix representing active constraint segments.
            curr_dq_grid: 3D array of DQ currents with shape (dim, n_torq, n_omega),
                          where dim is the number of DQ components (e.g., 4 for 5-phase, 8 for 9-phase).
            k_skip: Optional factor to downsample the grid (e.g., take every k-th point).
        """
        self.torq: np.ndarray = np.array(torq).flatten()
        self.omega: np.ndarray = np.array(omega).flatten()
        self.segments: np.ndarray = np.array(segments)
        self.curr_dq_grid: np.ndarray = np.array(curr_dq_grid)

        self._check_dimensions()

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
            ValueError: If any grid shape does not match (n_torq, n_omega).
        """
        n_torq = len(self.torq)
        n_omega = len(self.omega)

        expected_shape = (n_torq, n_omega)

        if self.segments.shape != expected_shape:
            raise ValueError(
                f"Segments shape {self.segments.shape} != expected {expected_shape}"
            )

        if self.curr_dq_grid.ndim != 3 or self.curr_dq_grid.shape[1:] != expected_shape:
            raise ValueError(
                f"curr_dq_grid shape {self.curr_dq_grid.shape} must be (dim, {n_torq}, {n_omega})."
            )

        if np.any(self.omega < 0):
            warnings.warn(
                "Negative speeds detected in grid data. Ensure reverse rotation is intended.",
                UserWarning,
                stacklevel=2,
            )

    def select_k(self, k_skip: int) -> None:
        """
        Subsamples the torque/speed vectors and all associated matrices.

        Args:
            k_skip: The step size for slicing the arrays.
        """
        self.omega = self.omega[::k_skip]
        self.torq = self.torq[::k_skip]

        self.segments = self.segments[::k_skip, ::k_skip]

        self.curr_dq_grid = self.curr_dq_grid[:, ::k_skip, ::k_skip]
