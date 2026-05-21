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

    def get_max_vals(self, omega: float, curr_dq: np.ndarray) -> tuple[float, float, float, float]:
        """
        Computes peak phase current and peak phase voltage for the given operating point.

        Returns a 4-tuple ``(curr_peak, 0.0, volt_peak, 0.0)``. The second
        and fourth slots are retained for backward API compatibility with
        callers that unpack four values; they no longer carry an alignment
        angle. Type I vs. Type II classification of the waveform is done
        directly on the time-domain waveform by ``count_peaks``.

        Args:
            omega: Electrical speed [rad/s].
            curr_dq: DQ current vector.

        Returns:
            Tuple: ``(curr_peak, 0.0, volt_peak, 0.0)``.
        """
        volt_ph, _, _ = self.get_volt_ph(omega, curr_dq)
        curr_ph = self.get_curr_ph(omega, curr_dq)
        curr_peak = float(np.max(np.abs(curr_ph)))
        volt_peak = float(np.max(np.abs(volt_ph)))
        return curr_peak, 0.0, volt_peak, 0.0

    def count_peaks(self, omega: float, curr_dq: np.ndarray, rel_tol: float = 1e-3) -> tuple[int, int]:
        """
        Strict active-set classification by direct local-maxima counting
        on the phase waveform. Per polarity, counts strict same-sign local
        extrema whose magnitude reaches the limit within ``rel_tol``
        (relative). The maximum across phases is returned for both
        current and voltage, capped at 2:

            0 : no local maximum reaches the limit
            1 : exactly one local extremum per polarity reaches the limit
                (Type I waveform, single peak per half-cycle)
            2 : two or more same-polarity local extrema reach the limit
                (Type II waveform, flat-top synthesis)

        Args:
            omega: Electrical speed [rad/s].
            curr_dq: DQ current vector.
            rel_tol: Relative tolerance on the peak magnitude.

        Returns:
            Tuple[int, int]: ``(n_curr_peaks, n_volt_peaks)``.

        Notes:
            Earlier versions of this method classified Type I/II via a
            dq-angle alignment heuristic. That heuristic was unreliable
            for two reasons: (a) it compared an angle in radians against
            ``tol * val_max`` in amperes/volts (dimensional mismatch),
            and (b) ``arctan2(0, 0)`` returns 0, so cells with no
            harmonic content were classified by an irrelevant computed
            value. The waveform-based detector here matches the
            active-set definition directly.
        """
        curr_ph = self.get_curr_ph(omega, curr_dq)
        volt_ph, _, _ = self.get_volt_ph(omega, curr_dq)
        n_curr_peaks = self._count_waveform_peaks_at_limit(curr_ph, self.machine.curr_max, rel_tol)
        n_volt_peaks = self._count_waveform_peaks_at_limit(volt_ph, self.machine.volt_max, rel_tol)
        return n_curr_peaks, n_volt_peaks

    @staticmethod
    def _count_waveform_peaks_at_limit(waveform: np.ndarray, limit: float, rel_tol: float) -> int:
        """
        Counts polarity-consistent local extrema of a 1D or 2D phase
        waveform that reach ``limit`` within ``rel_tol`` (relative).

        Args:
            waveform: shape ``(n_theta+1,)`` for a single phase or
                ``(n_phases, n_theta+1)`` stacked across phases. Per
                phase, looks at strict same-sign local extrema and counts
                those within ``limit * (1 - rel_tol)`` of the limit.
            limit: physical limit (``I_max`` or ``V_max``).
            rel_tol: relative tolerance.

        Returns:
            ``min(max_across_phases, 2)``. 0 if no peak reaches the
            limit anywhere; 1 for Type I (single same-polarity peak per
            phase); 2 for Type II (two or more same-polarity peaks).
        """
        if waveform.ndim == 1:
            waveform = waveform[np.newaxis, :]
        thresh = limit * (1.0 - rel_tol)
        n_max_global = 0
        for ph in range(waveform.shape[0]):
            w = waveform[ph, :-1]  # drop the wrap sample
            if w.size < 3:
                continue
            is_pmax = (w[1:-1] > w[:-2]) & (w[1:-1] > w[2:]) & (w[1:-1] > 0)
            n_pos = int(np.sum(w[1:-1][is_pmax] >= thresh))
            is_nmin = (w[1:-1] < w[:-2]) & (w[1:-1] < w[2:]) & (w[1:-1] < 0)
            n_neg = int(np.sum(-w[1:-1][is_nmin] >= thresh))
            n_max_global = max(n_max_global, n_pos, n_neg)
        return min(n_max_global, 2)
