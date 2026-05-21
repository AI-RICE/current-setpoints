"""
Induction-motor variant of ``Transform`` that uses the IM voltage
equation

    v_s = R_s i_s + omega_s * J * (L_s + L_mu * k_ir(omega_r)) i_s,

where ``omega_r = (R_r^1 / L_r^1) * i_sd^1 / i_sq^1`` is the
rotor-flux-orientation slip (Laksar et al., IM_TIA, eq. wr_opt) and
``k_ir`` is the rotor-current coupling matrix (im_coupling.k_ir_matrix).

The structural difference from the PMSM ``Transform`` is that the
voltage operator depends not only on the stator electrical speed
``omega`` (= omega_s) but also on the current vector through the
slip. The per-phase basis (DQ-to-phase trig matrix) is identical to
the PMSM case and uses ``machine.n_harmonics`` harmonics.
"""
from __future__ import annotations

import numpy as np

from ..parameters.im_machine import BaseMachineIM
from ..parameters.im_coupling import k_ir_matrix, slip_from_dq_foc


class IMTransform:
    """
    Mirrors the public surface of ``Transform`` so the existing
    optimizer / constraints / pointwise helpers consume it without
    modification.

    Differences from PMSM ``Transform``:
      * ``get_volt_dq``/``get_volt_ph`` recompute the voltage operator
        on every call because ``omega_r`` depends on ``curr_dq``.
      * No flux provider: BEMF is zero (no PM).
      * ``add_volt_0`` is accepted only as ``False`` for now; the
        SVPWM-extended modulation hooks of the IM_TIA / Vancini story
        are deferred.
    """

    def __init__(self, machine: BaseMachineIM, add_volt_0: bool = False, n_theta: int = 700) -> None:
        self.machine: BaseMachineIM = machine
        self.n_phases: int = machine.n_phases
        self.dim: int = machine.dim
        self.add_volt_0: bool = add_volt_0

        if add_volt_0:
            raise NotImplementedError("Zero-sequence injection not yet implemented for IMTransform")

        if n_theta <= 0:
            raise ValueError("n_theta must be positive")
        n_theta = round(n_theta / (2 * self.n_phases)) * 2 * self.n_phases
        if n_theta < 2 * self.n_phases:
            raise ValueError(
                f"n_theta is too small after rounding to a multiple of 2*n_phases "
                f"({2 * self.n_phases}). Increase the requested n_theta to at least "
                f"{2 * self.n_phases}."
            )
        self.vec_theta: np.ndarray = np.linspace(0, 2 * np.pi, n_theta + 1)
        self._phase_shift_samples: int = n_theta // self.n_phases

        # Phase-domain basis matrix (cos h*theta, -sin h*theta) for h in
        # the odd harmonics tracked by the machine. Identical to PMSM.
        cols = []
        for i in range(machine.n_harmonics):
            h = 2 * i + 1
            cols.append(np.cos(h * self.vec_theta))
            cols.append(-np.sin(h * self.vec_theta))
        self.mat_dq_to_ph: np.ndarray = np.column_stack(cols)

        # Cached state from the last call. omega is stator electrical
        # speed; omega_r is recomputed every call via the slip law.
        self.omega: float | None = None

    # ------------------------------------------------------------------
    # Voltage operator construction: U(omega_s, omega_r) and BEMF
    # ------------------------------------------------------------------

    def _build_volt_dq_operator(self, omega: float, omega_r: float) -> np.ndarray:
        """
        Returns the dq-frame voltage operator U(omega_s, omega_r):

            U = R_s + omega_s * J * (L_s + L_mu * k_ir(omega_r))

        With ``omega_r = 0`` this reduces to the conventional
        synchronous-speed inductive operator R_s + omega_s J L_s.
        """
        K = k_ir_matrix(omega_r, self.machine)
        L_eff = self.machine.L_s + self.machine.L_mu @ K
        return self.machine.R_stat + omega * self.machine.mat_crossc @ L_eff

    # ------------------------------------------------------------------
    # Public API matching PMSM Transform
    # ------------------------------------------------------------------

    def get_curr_ph(self, omega: float, curr_dq: np.ndarray) -> np.ndarray:
        """
        Phase-current waveform i_phase(theta) = (mat_dq_to_ph) @ i_s.
        Independent of omega; omega is accepted for API parity only.
        """
        self.omega = omega
        return self.mat_dq_to_ph @ curr_dq

    def get_volt_dq(self, omega: float, curr_dq: np.ndarray) -> np.ndarray:
        """
        DQ-frame stator voltage at the given operating point. Slip
        ``omega_r`` is derived internally from ``curr_dq`` via the
        rotor-flux-orientation rule.
        """
        self.omega = omega
        omega_r = slip_from_dq_foc(curr_dq, self.machine)
        U = self._build_volt_dq_operator(omega, omega_r)
        return U @ curr_dq

    def get_volt_ph(self, omega: float, curr_dq: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Phase-voltage waveform. Returns the 3-tuple
        ``(volt_ph, volt_0, volt_raw)`` for API parity with the PMSM
        path; ``volt_0`` is zero and ``volt_ph == volt_raw`` while
        SVPWM injection remains disabled.
        """
        self.omega = omega
        omega_r = slip_from_dq_foc(curr_dq, self.machine)
        U = self._build_volt_dq_operator(omega, omega_r)
        volt_dq = U @ curr_dq
        volt_raw = self.mat_dq_to_ph @ volt_dq
        return volt_raw, np.zeros_like(volt_raw), volt_raw

    # ------------------------------------------------------------------
    # Active-set regime classification.
    #
    # The PMSM Transform classifies Type I vs Type II by computing a
    # dq-angle alignment heuristic. That heuristic is unreliable for the
    # IM case for two reasons:
    #   1. dimensional mismatch -- the alignment angle (radians) is
    #      compared to a current/voltage tolerance (amperes/volts);
    #   2. undefined dq angle when i_3 = 0 (arctan2(0,0) returns 0, so
    #      pure-fundamental cells are classified by an irrelevant value).
    # Instead, we classify directly by counting positive (and negative)
    # local maxima of the actual phase waveform that reach the limit
    # within a relative tolerance. This matches the active-set
    # definition in App. A.6 of the IM-TTE paper draft.
    # ------------------------------------------------------------------

    def get_max_vals(self, omega: float, curr_dq: np.ndarray) -> tuple[float, float, float, float]:
        """
        Returns ``(curr_peak, curr_ang_diff, volt_peak, volt_ang_diff)``
        with the alignment fields kept for backward API parity. Only
        the peak values are used by ``count_peaks``.
        """
        volt_ph, _, _ = self.get_volt_ph(omega, curr_dq)
        curr_ph = self.get_curr_ph(omega, curr_dq)
        curr_peak = float(np.max(np.abs(curr_ph)))
        volt_peak = float(np.max(np.abs(volt_ph)))
        return curr_peak, 0.0, volt_peak, 0.0

    def count_peaks(self, omega: float, curr_dq: np.ndarray, rel_tol: float = 1e-3) -> tuple[int, int]:
        """
        Strict active-set classification by direct local-maxima
        counting on the waveform.

        Returns ``(n_curr, n_volt)`` each in ``{0, 1, 2}``:
          * 0 if no local maximum of the (per-phase) waveform reaches
            the limit within ``rel_tol`` relative tolerance;
          * 1 if exactly one local extremum per polarity reaches the
            limit (Type I shape);
          * 2 if two or more local extrema of the same polarity reach
            the limit (Type II shape).

        Across phases, the maximum count is returned. ``rel_tol`` is a
        relative tolerance on the peak magnitude; the previous absolute
        ``tol`` argument is replaced because SLSQP-converged solutions
        often hit the limit with a small constraint-satisfaction
        residual that is naturally relative, not absolute.
        """
        curr_ph = self.get_curr_ph(omega, curr_dq)
        volt_ph, _, _ = self.get_volt_ph(omega, curr_dq)
        n_curr = self._count_waveform_peaks_at_limit(curr_ph, self.machine.curr_max, rel_tol)
        n_volt = self._count_waveform_peaks_at_limit(volt_ph, self.machine.volt_max, rel_tol)
        return n_curr, n_volt

    @staticmethod
    def _count_waveform_peaks_at_limit(waveform: np.ndarray, limit: float, rel_tol: float) -> int:
        """
        For a (n_phases, n_theta+1) (or (n_theta+1,)) phase-waveform
        array, returns the maximum across phases of the count of
        polarity-consistent local extrema that reach ``limit`` within
        ``rel_tol``, capped at 2.

        Per polarity, a local extremum at the limit is identified by:
            * a strict same-sign local maximum/minimum
            * magnitude >= limit * (1 - rel_tol)
        The Type I / Type II distinction is then the count per polarity,
        capped at 2.
        """
        if waveform.ndim == 1:
            waveform = waveform[np.newaxis, :]
        thresh = limit * (1.0 - rel_tol)
        n_max_global = 0
        for ph in range(waveform.shape[0]):
            w = waveform[ph, :-1]  # drop the wrap sample
            if w.size < 3:
                continue
            # Positive local maxima
            is_pmax = (w[1:-1] > w[:-2]) & (w[1:-1] > w[2:]) & (w[1:-1] > 0)
            n_pos = int(np.sum(w[1:-1][is_pmax] >= thresh))
            # Negative local minima (peaks of -w)
            is_nmin = (w[1:-1] < w[:-2]) & (w[1:-1] < w[2:]) & (w[1:-1] < 0)
            n_neg = int(np.sum(-w[1:-1][is_nmin] >= thresh))
            n_max_global = max(n_max_global, n_pos, n_neg)
        return min(n_max_global, 2)
