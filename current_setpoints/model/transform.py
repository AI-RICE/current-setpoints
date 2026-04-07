import numpy as np
from typing import Any, Tuple
from utils._types import MachineProtocol

class Transform:
    """
    Handles reference frame transformations (DQ to Phase) and voltage/current
    calculations for multiphase machines, including zero-sequence injection.
    """

    def __init__(
        self, machine: MachineProtocol, omega: float, add_volt_0: bool, n_theta: int = 700
    ) -> None:
        """
        Initializes the transform instance with machine parameters and resolution.

        Args:
            machine: Object containing n_phases, R_stat, L_stat, and flux_volt.
            omega: Initial electrical speed in rad/s.
            add_volt_0: Boolean flag to enable zero-sequence (SVPWM) voltage injection.
            n_theta: Angular resolution for phase mapping.
        """
        self.n_phases: int = machine.n_phases
        self.add_volt_0: bool = add_volt_0

        # Validate theta resolution
        if n_theta <= 0:
            raise ValueError("n_theta must be positive")

        n_theta = round(n_theta / (2 * machine.n_phases)) * 2 * machine.n_phases
        self.vec_theta: np.ndarray = np.linspace(0, 2 * np.pi, n_theta + 1)

        self.compute_matrices_init(machine)
        self.compute_matrices(omega)

    def compute_matrices_init(self, machine: Any) -> None:
        """
        Computes speed-independent matrices for the DQ-to-Phase transformation.

        Args:
            machine: Machine object containing physical constant matrices.
        """
        self.mat_curr_dq_to_volt_dq_fixed: np.ndarray = machine.R_stat @ np.eye(4)
        self.mat_curr_dq_to_volt_dq_omega: np.ndarray = (
            machine.mat_crossc @ machine.L_stat
        )
        self.vec_volt_bemf_dq_omega: np.ndarray = machine.mat_crossc @ machine.flux_volt

        cos_theta = np.cos(self.vec_theta)
        sin_theta = np.sin(self.vec_theta)
        cos_3theta = np.cos(3 * self.vec_theta)
        sin_3theta = np.sin(3 * self.vec_theta)
        self.mat_dq_to_ph: np.ndarray = np.array(
            [cos_theta, -sin_theta, cos_3theta, -sin_3theta]
        ).T

    def compute_matrices(self, omega: float) -> None:
        """
        Updates speed-dependent matrices (impedance and Back-EMF).

        Args:
            omega: Electrical speed in rad/s.
        """
        self.omega: float = omega
        self.mat_curr_dq_to_volt_dq: np.ndarray = (
            self.mat_curr_dq_to_volt_dq_fixed
            + omega * self.mat_curr_dq_to_volt_dq_omega
        )
        self.vec_volt_bemf_dq: np.ndarray = omega * self.vec_volt_bemf_dq_omega
        self.mat_curr_dq_to_volt_ph: np.ndarray = (
            self.mat_dq_to_ph @ self.mat_curr_dq_to_volt_dq
        )
        self.vec_volt_bemf_ph: np.ndarray = self.mat_dq_to_ph @ self.vec_volt_bemf_dq

    def set_omega(self, omega: float) -> None:
        """
        Sets a new operating speed and updates relevant transformation matrices.

        Args:
            omega: Electrical speed in rad/s.
        """
        if not isinstance(omega, (int, float)):
            raise TypeError(f"Omega must be a real scalar, got {type(omega)}")
        self.compute_matrices(float(omega))

    def get_curr_ph(self, vec_curr_dq: np.ndarray) -> np.ndarray:
        """
        Transforms DQ currents into phase current time-series.

        Args:
            vec_curr_dq: 4-element current vector [id1, iq1, id3, iq3].

        Returns:
            np.ndarray: Vector of currents in the phase domain.
        """
        if len(vec_curr_dq) != 4:
            raise ValueError("Current vector must be length 4 (id1, iq1, id3, iq3)")
        return self.mat_dq_to_ph @ vec_curr_dq

    def get_volt_dq(self, vec_curr_dq: np.ndarray) -> np.ndarray:
        """
        Calculates the DQ voltage vector based on current and speed.

        Args:
            vec_curr_dq: 4-element current vector.

        Returns:
            np.ndarray: 4-element voltage vector.
        """
        return self.mat_curr_dq_to_volt_dq @ vec_curr_dq + self.vec_volt_bemf_dq

    def get_volt_ph(
        self, vec_curr_dq: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates phase voltages, including zero-sequence components if enabled.

        Args:
            vec_curr_dq: 4-element current vector.

        Returns:
            Tuple: (final_phase_voltage, zero_sequence_voltage, raw_phase_voltage).
        """
        if vec_curr_dq.ndim != 1 or len(vec_curr_dq) != 4:
            # Critical check for optimization loops
            raise ValueError(
                f"Input current vector shape mismatch: {vec_curr_dq.shape}"
            )

        vec_volt_raw = self.mat_curr_dq_to_volt_ph @ vec_curr_dq + self.vec_volt_bemf_ph
        if self.add_volt_0:
            mat_volt_res = vec_volt_raw[:-1].reshape(-1, self.n_phases)
            vec_volt_0 = -0.5 * (
                np.min(mat_volt_res, axis=1) + np.max(mat_volt_res, axis=1)
            )
            vec_volt_ph = mat_volt_res
            vec_volt_ph = vec_volt_ph.flatten()
            vec_volt_ph = np.append(vec_volt_ph, vec_volt_ph[0])
            vec_volt_0 = np.repeat(vec_volt_0, self.n_phases)
            vec_volt_0 = np.append(vec_volt_0, vec_volt_0[0])
        else:
            vec_volt_ph = vec_volt_raw
            vec_volt_0 = np.zeros_like(vec_volt_raw)
        return vec_volt_ph, vec_volt_0, vec_volt_raw

    def get_max_vals(
        self, vec_curr_dq: np.ndarray
    ) -> Tuple[float, float, np.ndarray, float, float, float, float, float]:
        """
        Computes peak values, angles, and RMS metrics for current and voltage.

        Args:
            vec_curr_dq: 4-element current vector.

        Returns:
            Tuple: (curr_peak, curr_ang_diff, vec_volt_dq, volt_peak, volt_ang_diff,
                    volt_raw_peak, volt_0_rms, volt_0_peak).
        """
        vec_volt_ph, vec_volt_0, vec_volt_raw = self.get_volt_ph(vec_curr_dq)
        curr_peak = np.max(np.abs(self.get_curr_ph(vec_curr_dq)))
        volt_peak = np.max(np.abs(vec_volt_ph))
        volt_raw_peak = np.max(np.abs(vec_volt_raw))
        volt_0_rms = np.sqrt(np.mean(vec_volt_0**2))
        volt_0_peak = np.max(np.abs(vec_volt_0))
        vec_volt_dq = self.get_volt_dq(vec_curr_dq)

        # Phase calculations
        curr_ang_1 = np.arctan2(vec_curr_dq[1], vec_curr_dq[0])
        curr_ang_3 = np.arctan2(vec_curr_dq[3], vec_curr_dq[2])
        curr_ang_diff = np.abs(
            np.mod(curr_ang_1 * 3 + np.pi, 2 * np.pi) - np.mod(curr_ang_3, 2 * np.pi)
        )

        volt_ang_1 = np.arctan2(vec_volt_dq[1], vec_volt_dq[0])
        volt_ang_3 = np.arctan2(vec_volt_dq[3], vec_volt_dq[2])
        volt_ang_diff = np.abs(
            np.mod(volt_ang_1 * 3 + np.pi, 2 * np.pi) - np.mod(volt_ang_3, 2 * np.pi)
        )

        return (
            curr_peak,
            curr_ang_diff,
            vec_volt_dq,
            volt_peak,
            volt_ang_diff,
            volt_raw_peak,
            volt_0_rms,
            volt_0_peak,
        )

    def count_peaks(
        self, vec_curr_dq: np.ndarray, machine: Any, tol: float = 1e-4
    ) -> Tuple[int, int]:
        """
        Determines the number of active peaks hitting physical limits for current and voltage.

        Args:
            vec_curr_dq: 4-element current vector.
            machine: Machine object containing curr_max and volt_max.
            tol: Numerical tolerance for comparing peaks to limits.

        Returns:
            Tuple[int, int]: Number of current peaks (0, 1, or 2), number of voltage peaks.
        """
        curr_peak, curr_ang_diff, _, volt_peak, volt_ang_diff, _, _, _ = (
            self.get_max_vals(vec_curr_dq)
        )
        n_curr_peaks = self._count_peaks_helper(
            curr_peak, machine.curr_max, curr_ang_diff, tol
        )
        n_volt_peaks = self._count_peaks_helper(
            volt_peak, machine.volt_max, volt_ang_diff, tol
        )
        return n_curr_peaks, n_volt_peaks

    def _count_peaks_helper(
        self, value: float, val_max: float, angle_diff: float, tol: float
    ) -> int:
        """
        Internal logic to determine if 0, 1, or 2 peaks are present based on limit proximity.

        Args:
            value: The calculated peak value.
            val_max: The physical limit.
            angle_diff: The calculated angular difference between harmonics.
            tol: Numerical tolerance.

        Returns:
            int: 0 (not at limit), 1 (single peak), or 2 (two peaks at limit).
        """
        if np.abs(value - val_max) < tol:
            if angle_diff < tol * val_max:
                return 2
            else:
                return 1
        else:
            return 0
