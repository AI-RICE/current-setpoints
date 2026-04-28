import numpy as np

from ..data import BaseMachine


class Transform:
    """
    Handles reference frame transformations (DQ to Phase) and voltage/current
    calculations for multiphase machines, including zero-sequence injection.
    Supports dynamic flux map updates.
    """

    def __init__(
        self, machine: BaseMachine, omega: float, add_volt_0: bool, n_theta: int = 700
    ) -> None:
        """
        Initializes the transform instance with machine parameters and resolution.

        Args:
            machine: Object containing n_phases, R_stat, L_stat, flux_volt, and update_state.
            omega: Initial electrical speed in rad/s.
            add_volt_0: Boolean flag to enable zero-sequence (SVPWM) voltage injection.
            n_theta: Angular resolution for phase mapping. Internally rounded to a
                multiple of (2 * n_phases) so that an integer number of samples
                corresponds to a phase shift of 2*pi/n_phases.
        """
        self.machine: BaseMachine = machine
        self.n_phases: int = machine.n_phases
        self.dim: int = self.n_phases - 1
        self.add_volt_0: bool = add_volt_0

        if n_theta <= 0:
            raise ValueError("n_theta must be positive")

        n_theta = round(n_theta / (2 * machine.n_phases)) * 2 * machine.n_phases
        if n_theta < 2 * machine.n_phases:
            raise ValueError(
                f"n_theta is too small after rounding to a multiple of "
                f"2*n_phases ({2 * machine.n_phases}). Increase the requested "
                f"n_theta to at least {2 * machine.n_phases}."
            )
        self.vec_theta: np.ndarray = np.linspace(0, 2 * np.pi, n_theta + 1)

        self._phase_shift_samples: int = n_theta // self.n_phases

        self.compute_matrices_init()
        self.compute_matrices(omega)

    def compute_matrices_init(self) -> None:
        """
        Computes speed-independent matrices for the DQ-to-Phase transformation.
        Note: BEMF is no longer precomputed here because flux is dynamic.
        """
        self.mat_curr_dq_to_volt_dq_fixed: np.ndarray = self.machine.R_stat.copy()
        self.mat_curr_dq_to_volt_dq_omega: np.ndarray = (
            self.machine.mat_crossc @ self.machine.L_stat
        )

        cols = []
        for i in range(self.dim // 2):
            h = 2 * i + 1  # Harmonic number: 1, 3, 5, 7...
            cols.append(np.cos(h * self.vec_theta))
            cols.append(-np.sin(h * self.vec_theta))

        self.mat_dq_to_ph: np.ndarray = np.column_stack(cols)

    def compute_matrices(self, omega: float) -> None:
        """
        Updates speed-dependent impedance matrices.
        Note: BEMF is calculated on the fly in the getters.

        Args:
            omega: Electrical speed in rad/s.
        """
        self.omega: float = omega
        self.mat_curr_dq_to_volt_dq: np.ndarray = (
            self.mat_curr_dq_to_volt_dq_fixed
            + omega * self.mat_curr_dq_to_volt_dq_omega
        )
        self.mat_curr_dq_to_volt_ph: np.ndarray = (
            self.mat_dq_to_ph @ self.mat_curr_dq_to_volt_dq
        )

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
            vec_curr_dq: DQ current vector.

        Returns:
            np.ndarray: Vector of currents in the phase domain.
        """
        if len(vec_curr_dq) != self.dim:
            raise ValueError(f"Current vector must be length {self.dim}")
        return self.mat_dq_to_ph @ vec_curr_dq

    def _get_volt_dq_internal(self, vec_curr_dq: np.ndarray) -> np.ndarray:
        """Computes DQ voltage. Caller is responsible for state freshness."""
        vec_volt_bemf_dq = self.omega * (
            self.machine.mat_crossc @ self.machine.flux_volt
        )
        return self.mat_curr_dq_to_volt_dq @ vec_curr_dq + vec_volt_bemf_dq

    def _get_volt_ph_internal(
        self, vec_curr_dq: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Computes phase voltage tuple. Caller is responsible for state freshness."""
        if vec_curr_dq.ndim != 1 or len(vec_curr_dq) != self.dim:
            raise ValueError(
                f"Input current vector shape mismatch: {vec_curr_dq.shape}. Expected ({self.dim},)"
            )

        vec_volt_bemf_dq = self.omega * (
            self.machine.mat_crossc @ self.machine.flux_volt
        )
        vec_volt_bemf_ph = self.mat_dq_to_ph @ vec_volt_bemf_dq
        vec_volt_raw = self.mat_curr_dq_to_volt_ph @ vec_curr_dq + vec_volt_bemf_ph

        if self.add_volt_0:
            n_t = self.vec_theta.size - 1
            step = self._phase_shift_samples
            raw = vec_volt_raw[:-1]

            time_idx = np.arange(n_t)
            shift_idx = np.arange(self.n_phases) * step
            indices = (time_idx[None, :] - shift_idx[:, None]) % n_t
            mat_phases = raw[indices]  # (n_phases, n_t)

            vec_volt_0_t = -0.5 * (mat_phases.min(axis=0) + mat_phases.max(axis=0))

            vec_volt_ph = raw + vec_volt_0_t
            vec_volt_ph = np.append(vec_volt_ph, vec_volt_ph[0])
            vec_volt_0 = np.append(vec_volt_0_t, vec_volt_0_t[0])
        else:
            vec_volt_ph = vec_volt_raw
            vec_volt_0 = np.zeros_like(vec_volt_raw)
        return vec_volt_ph, vec_volt_0, vec_volt_raw

    def get_volt_dq(self, vec_curr_dq: np.ndarray) -> np.ndarray:
        """
        Calculates the DQ voltage vector based on current, speed, and dynamic flux.

        Args:
            vec_curr_dq: DQ current vector.

        Returns:
            np.ndarray: DQ voltage vector.
        """
        self.machine.update_state(self.omega, vec_curr_dq)
        return self._get_volt_dq_internal(vec_curr_dq)

    def get_volt_ph(
        self, vec_curr_dq: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates phase voltages, including zero-sequence components if enabled.

        When ``add_volt_0`` is True, applies min-max zero-sequence injection
        (commonly known as SVPWM) to the phase-A voltage time-series. The
        injected common-mode signal is computed across all n_phases at each
        time sample (using the symmetry that phase x at time k equals phase A
        sampled at time k - x * n_theta/n_phases) so that the resulting
        vec_volt_ph genuinely has reduced peak magnitude vs the raw waveform.

        Args:
            vec_curr_dq: DQ current vector.

        Returns:
            Tuple: (final_phase_voltage, zero_sequence_voltage, raw_phase_voltage).
            All three arrays share the same length (n_theta + 1, last sample is
            a periodic wrap of the first).
        """
        self.machine.update_state(self.omega, vec_curr_dq)
        return self._get_volt_ph_internal(vec_curr_dq)

    @staticmethod
    def _harmonic_alignment_diff(ang_fundamental: float, ang_h: float, h: int) -> float:
        """
        Returns the circular distance, in [0, pi], between (h * ang_fundamental + pi)
        and ang_h. The condition `h*phi_1 + pi == phi_h (mod 2*pi)` corresponds to
        flat-top alignment between the fundamental and the h-th harmonic, in which
        the h-th harmonic subtracts maximally at the fundamental's peak.
        """
        delta = (h * ang_fundamental + np.pi - ang_h) % (2 * np.pi)
        return float(min(delta, 2 * np.pi - delta))

    def _alignment_diff_all_harmonics(self, vec_dq: np.ndarray) -> float:
        """
        Worst-case (max) alignment deviation across all odd harmonics above the
        fundamental present in the DQ vector. Generalises the original 5-phase
        1st-vs-3rd alignment check to any number of phases:

            5-phase  (dim=4): max over {h=3}
            7-phase  (dim=6): max over {h=3, 5}
            9-phase  (dim=8): max over {h=3, 5, 7}

        Returns 0.0 for machines without harmonics above the fundamental.
        """
        if self.dim < 4:
            return 0.0
        ang_1 = np.arctan2(vec_dq[1], vec_dq[0])
        diffs = []
        for i in range(1, self.dim // 2):
            h = 2 * i + 1  # 3, 5, 7, ...
            ang_h = np.arctan2(vec_dq[2 * i + 1], vec_dq[2 * i])
            diffs.append(self._harmonic_alignment_diff(ang_1, ang_h, h))
        return max(diffs)

    def get_max_vals(
        self, vec_curr_dq: np.ndarray
    ) -> tuple[float, float, np.ndarray, float, float, float, float, float]:
        """
        Computes peak values, angles, and RMS metrics for current and voltage.

        Args:
            vec_curr_dq: DQ current vector.

        Returns:
            Tuple: (curr_peak, curr_ang_diff, vec_volt_dq, volt_peak, volt_ang_diff,
                    volt_raw_peak, volt_0_rms, volt_0_peak).
        """
        self.machine.update_state(self.omega, vec_curr_dq)

        vec_volt_ph, vec_volt_0, vec_volt_raw = self._get_volt_ph_internal(vec_curr_dq)
        vec_volt_dq = self._get_volt_dq_internal(vec_curr_dq)

        curr_peak = np.max(np.abs(self.get_curr_ph(vec_curr_dq)))
        volt_peak = np.max(np.abs(vec_volt_ph))
        volt_raw_peak = np.max(np.abs(vec_volt_raw))
        volt_0_rms = np.sqrt(np.mean(vec_volt_0**2))
        volt_0_peak = np.max(np.abs(vec_volt_0))

        curr_ang_diff = self._alignment_diff_all_harmonics(vec_curr_dq)
        volt_ang_diff = self._alignment_diff_all_harmonics(vec_volt_dq)

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
        self, vec_curr_dq: np.ndarray, machine: BaseMachine, tol: float = 1e-4
    ) -> tuple[int, int]:
        """
        Determines the number of active peaks hitting physical limits for current and voltage.

        Args:
            vec_curr_dq: DQ current vector.
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
