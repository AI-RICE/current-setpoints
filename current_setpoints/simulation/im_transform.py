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
    # Diagnostics: peak detection and regime classification, copied
    # verbatim from PMSM Transform so the active-set logic works
    # unchanged.
    # ------------------------------------------------------------------

    @staticmethod
    def _harmonic_alignment_diff(ang_fundamental: float, ang_h: float, h: int) -> float:
        delta = (h * ang_fundamental + np.pi - ang_h) % (2 * np.pi)
        return float(min(delta, 2 * np.pi - delta))

    def _alignment_diff_all_harmonics(self, vec_dq: np.ndarray) -> float:
        if self.dim < 4:
            return 0.0
        ang_1 = np.arctan2(vec_dq[1], vec_dq[0])
        diffs = []
        for i in range(1, self.dim // 2):
            h = 2 * i + 1
            ang_h = np.arctan2(vec_dq[2 * i + 1], vec_dq[2 * i])
            diffs.append(self._harmonic_alignment_diff(ang_1, ang_h, h))
        return max(diffs)

    def get_max_vals(self, omega: float, curr_dq: np.ndarray) -> tuple[float, float, float, float]:
        volt_ph, _, _ = self.get_volt_ph(omega, curr_dq)
        curr_ph = self.get_curr_ph(omega, curr_dq)
        volt_dq = self.get_volt_dq(omega, curr_dq)
        curr_peak = float(np.max(np.abs(curr_ph)))
        volt_peak = float(np.max(np.abs(volt_ph)))
        curr_ang_diff = self._alignment_diff_all_harmonics(curr_dq)
        volt_ang_diff = self._alignment_diff_all_harmonics(volt_dq)
        return curr_peak, curr_ang_diff, volt_peak, volt_ang_diff

    def count_peaks(self, omega: float, curr_dq: np.ndarray, tol: float = 1e-4) -> tuple[int, int]:
        curr_peak, curr_ang_diff, volt_peak, volt_ang_diff = self.get_max_vals(omega, curr_dq)
        n_curr_peaks = self._count_peaks_helper(curr_peak, self.machine.curr_max, curr_ang_diff, tol)
        n_volt_peaks = self._count_peaks_helper(volt_peak, self.machine.volt_max, volt_ang_diff, tol)
        return n_curr_peaks, n_volt_peaks

    def _count_peaks_helper(self, value: float, val_max: float, angle_diff: float, tol: float) -> int:
        if np.abs(value - val_max) < tol:
            return 2 if angle_diff < tol * val_max else 1
        return 0
