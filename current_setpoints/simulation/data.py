import warnings

import numpy as np


class MachineData:
    """
    Container class for motor grid data. Handles dimensionality validation,
    subsampling, and coordinate grid generation for motor performance mapping
    across any number of phases/harmonics.
    """

    def __init__(
        self,
        torq: np.ndarray | list[float],
        omega: np.ndarray | list[float],
        segments: np.ndarray,
        curr_dq_grid: np.ndarray,
        k_skip: int | None = None,
    ) -> None:
        """
        Initializes MachineData with torque/speed vectors and current component matrices.

        Args:
            torq: Torque axis values [Nm].
            omega: Speed axis values [mechanical RPM]. Typically produced by
                ``grid_to_data`` which converts the internal electrical-rad/s
                representation via ``const_mech_speed``.
            segments: 2D matrix of operating-regime labels, shape
                (n_torq, n_omega). Encoded as ``3 * n_volt_peaks + n_curr_peaks``
                (integer values 0-8). Unfilled cells are ``NaN``, so the
                array dtype is float.
            curr_dq_grid: 3D array of DQ currents, shape (dim, n_torq, n_omega),
                ordered ``[d_1, q_1, d_3, q_3, ...]`` along axis 0, where
                ``dim = n_phases - 1`` (e.g., 4 for 5-phase, 8 for 9-phase).
                Unfilled cells are ``NaN``.
            k_skip: Optional factor to downsample the grid (e.g., take every
                k-th point along both axes). ``None`` or ``1`` keeps every sample.
        """
        self.torq: np.ndarray = np.array(torq).flatten()
        self.omega: np.ndarray = np.array(omega).flatten()
        self.segments: np.ndarray = np.array(segments)
        self.curr_dq_grid: np.ndarray = np.array(curr_dq_grid)

        self._check_dimensions()

        if k_skip is not None and k_skip > 1:
            self.select_k(k_skip)

    def _check_dimensions(self) -> None:
        """
        Ensures all input arrays are compatible with torq and omega dimensions.

        Raises:
            ValueError: If torq/omega are empty or if any grid shape does not
                match (n_torq, n_omega).
        """
        n_torq = len(self.torq)
        n_omega = len(self.omega)

        if n_torq == 0 or n_omega == 0:
            raise ValueError(f"torq and omega must both be non-empty; got len(torq)={n_torq}, len(omega)={n_omega}.")

        expected_shape = (n_torq, n_omega)

        if self.segments.shape != expected_shape:
            raise ValueError(f"Segments shape {self.segments.shape} != expected {expected_shape}")

        if self.curr_dq_grid.ndim != 3 or self.curr_dq_grid.shape[1:] != expected_shape:
            raise ValueError(f"curr_dq_grid shape {self.curr_dq_grid.shape} must be (dim, {n_torq}, {n_omega}).")

        if np.any(self.omega < 0):
            warnings.warn(
                "Negative speeds detected in grid data. Ensure reverse rotation is intended.",
                UserWarning,
                stacklevel=2,
            )

    def select_k(self, k_skip: int) -> None:
        """
        Subsamples the torque/speed vectors and all associated matrices by
        taking every ``k_skip``-th element along both grid axes.

        Args:
            k_skip: The step size for slicing the arrays. Must be >= 1.
        """
        if not isinstance(k_skip, (int, np.integer)) or k_skip < 1:
            raise ValueError(f"k_skip must be a positive integer (>= 1), got {k_skip!r}.")
        if k_skip == 1:
            return

        self.omega = self.omega[::k_skip]
        self.torq = self.torq[::k_skip]
        self.segments = self.segments[::k_skip, ::k_skip]
        self.curr_dq_grid = self.curr_dq_grid[:, ::k_skip, ::k_skip]

    @property
    def torq_grid(self) -> np.ndarray:
        """Meshgrid of torque values, shape (n_torq, n_omega)."""
        grid, _ = np.meshgrid(self.torq, self.omega, indexing="ij")
        return grid

    @property
    def omega_grid(self) -> np.ndarray:
        """Meshgrid of speed values, shape (n_torq, n_omega)."""
        _, grid = np.meshgrid(self.torq, self.omega, indexing="ij")
        return grid

    @property
    def unique_segments(self) -> np.ndarray:
        """Sorted, integer-cast unique non-NaN segment labels present in the grid."""
        return np.unique(self.segments[~np.isnan(self.segments)]).astype(int)
