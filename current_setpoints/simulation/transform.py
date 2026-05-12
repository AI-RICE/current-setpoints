import numpy as np

from ..parameters import BaseMachine, Flux


class Transform:
    """
    Handles reference frame transformations (DQ to Phase) and voltage/current
    calculations for multiphase machines. Zero-sequence injection is planned
    but not yet implemented (see ``get_volt_ph``). Supports dynamic flux
    map updates.
    """

    def __init__(self, machine: BaseMachine, flux: Flux, add_volt_0: bool, n_theta: int = 700) -> None:
        """
        Initializes the transform instance with machine parameters and resolution.

        Args:
            machine: ``BaseMachine`` instance providing ``n_phases``,
                ``R_stat``, ``L_stat``, and ``mat_crossc``.
            flux: ``Flux`` provider returning ``(flux_volt, flux_torq)``
                vectors for a given operating point.
            add_volt_0: Boolean flag to enable zero-sequence (SVPWM) voltage
                injection. Currently raises ``NotImplementedError`` when True.
            n_theta: Angular resolution for phase mapping. Internally rounded to a
                multiple of (2 * n_phases) so that an integer number of samples
                corresponds to a phase shift of 2*pi/n_phases.
        """
        self.machine: BaseMachine = machine
        self.flux = flux
        self.n_phases: int = machine.n_phases
        self.dim: int = self.n_phases - 1
        self.omega: float | None = None
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

        self._compute_matrices_init()

    def _compute_matrices_init(self) -> None:
        """
        Computes speed-independent matrices for the DQ-to-Phase transformation.
        Note: BEMF is no longer precomputed here because flux is dynamic.
        """
        self.mat_curr_dq_to_volt_dq_fixed: np.ndarray = self.machine.R_stat.copy()
        self.mat_curr_dq_to_volt_dq_omega: np.ndarray = self.machine.mat_crossc @ self.machine.L_stat

        cols = []
        for i in range(self.dim // 2):
            h = 2 * i + 1  # Harmonic number: 1, 3, 5, 7...
            cols.append(np.cos(h * self.vec_theta))
            cols.append(-np.sin(h * self.vec_theta))

        self.mat_dq_to_ph: np.ndarray = np.column_stack(cols)

    def _set_omega(self, omega: float) -> None:
        """
        Sets a new operating speed and updates relevant transformation matrices.

        Args:
            omega: Electrical speed in rad/s.
        """

        if self.omega is None or omega != self.omega:
            self.omega = omega
            self.mat_curr_dq_to_volt_dq = (
                self.mat_curr_dq_to_volt_dq_fixed + self.omega * self.mat_curr_dq_to_volt_dq_omega
            )
            self.mat_curr_dq_to_volt_ph = self.mat_dq_to_ph @ self.mat_curr_dq_to_volt_dq

    def get_curr_ph(self, omega: float, curr_dq: np.ndarray) -> np.ndarray:
        """
        Transforms DQ currents into phase current time-series.

        Args:
            omega: Electrical speed [rad/s].
            curr_dq: DQ current vector.

        Returns:
            np.ndarray: Vector of currents in the phase domain.
        """

        self._set_omega(omega)
        return self.mat_dq_to_ph @ curr_dq

    def get_volt_dq(self, omega: float, curr_dq: np.ndarray) -> np.ndarray:
        """
        Calculates the DQ voltage vector based on current, speed, and dynamic flux.

        Args:
            omega: Electrical speed [rad/s].
            curr_dq: DQ current vector.

        Returns:
            np.ndarray: DQ voltage vector.
        """

        self._set_omega(omega)
        flux_volt, _ = self.flux.get_flux(omega, curr_dq)
        vec_volt_bemf_dq = self.omega * self.machine.mat_crossc @ flux_volt
        return self.mat_curr_dq_to_volt_dq @ curr_dq + vec_volt_bemf_dq

    def get_volt_ph(self, omega: float, curr_dq: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates phase voltages. Zero-sequence injection via ``add_volt_0``
        is not yet implemented and raises ``NotImplementedError`` when enabled.

        When implemented, ``add_volt_0=True`` will apply min-max zero-sequence
        injection (commonly known as SVPWM) to the phase-A voltage time-series.
        The injected common-mode signal will be computed across all n_phases at
        each time sample (using the symmetry that phase x at time k equals
        phase A sampled at time k - x * n_theta/n_phases) so that the resulting
        vec_volt_ph genuinely has reduced peak magnitude vs the raw waveform.

        Args:
            omega: Electrical speed [rad/s].
            curr_dq: DQ current vector.

        Returns:
            Tuple: (final_phase_voltage, zero_sequence_voltage, raw_phase_voltage).
            All three arrays share the same length (n_theta + 1, last sample is
            a periodic wrap of the first). With injection currently disabled,
            final_phase_voltage equals raw_phase_voltage and zero_sequence_voltage
            is the zero array.
        """

        self._set_omega(omega)
        flux_volt, _ = self.flux.get_flux(omega, curr_dq)
        volt_bemf_dq = self.omega * self.machine.mat_crossc @ flux_volt
        volt_bemf_ph = self.mat_dq_to_ph @ volt_bemf_dq
        volt_raw = self.mat_curr_dq_to_volt_ph @ curr_dq + volt_bemf_ph

        if self.add_volt_0:
            raise NotImplementedError("Zero-sequence injection not yet implemented")
        else:
            volt_ph = volt_raw
            volt_0 = np.zeros_like(volt_raw)
        return volt_ph, volt_0, volt_raw

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

    def get_max_vals(self, omega: float, curr_dq: np.ndarray) -> tuple[float, float, float, float]:
        """
        Computes peak phase current/voltage and the worst-case harmonic
        alignment angle for each.

        Args:
            omega: Electrical speed [rad/s].
            curr_dq: DQ current vector.

        Returns:
            Tuple: (curr_peak, curr_ang_diff, volt_peak, volt_ang_diff).
        """
        volt_ph, _, _ = self.get_volt_ph(omega, curr_dq)
        curr_ph = self.get_curr_ph(omega, curr_dq)
        volt_dq = self.get_volt_dq(omega, curr_dq)

        curr_peak = np.max(np.abs(curr_ph))
        volt_peak = np.max(np.abs(volt_ph))

        curr_ang_diff = self._alignment_diff_all_harmonics(curr_dq)
        volt_ang_diff = self._alignment_diff_all_harmonics(volt_dq)

        return curr_peak, curr_ang_diff, volt_peak, volt_ang_diff

    def count_peaks(self, omega: float, curr_dq: np.ndarray, tol: float = 1e-4) -> tuple[int, int]:
        """
        Determines the number of active peaks hitting physical limits for current and voltage.

        Args:
            omega: Electrical speed [rad/s].
            curr_dq: DQ current vector.
            tol: Numerical tolerance for comparing peaks to limits.

        Returns:
            Tuple[int, int]: Number of current peaks (0, 1, or 2), number of voltage peaks.
        """
        curr_peak, curr_ang_diff, volt_peak, volt_ang_diff = self.get_max_vals(omega, curr_dq)
        n_curr_peaks = self._count_peaks_helper(curr_peak, self.machine.curr_max, curr_ang_diff, tol)
        n_volt_peaks = self._count_peaks_helper(volt_peak, self.machine.volt_max, volt_ang_diff, tol)
        return n_curr_peaks, n_volt_peaks

    def _count_peaks_helper(self, value: float, val_max: float, angle_diff: float, tol: float) -> int:
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
