from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..parameters import BaseMachine, Flux


class BaseTransform(ABC):
    """
    Abstract foundation for DQ-to-phase transformations on multiphase
    machines. Holds the common DQ-side voltage equation and the
    harmonic-alignment / peak-counting utilities; subclasses supply the
    phase-mapping matrix and the phase-domain accessors that depend on
    which physical phases are healthy.

    Concrete subclasses:
        Transform                    - All phases healthy.
        TransformFault1              - One open phase.
        TransformFault2Adjacent      - Two adjacent open phases.
        TransformFault2NonAdjacent   - Two non-adjacent open phases.
    """

    def __init__(self, machine: BaseMachine, flux: Flux, n_theta: int = 700) -> None:
        """
        Initializes the shared transform state.

        Args:
            machine: ``BaseMachine`` instance providing ``n_phases``,
                ``R_stat``, ``L_stat``, and ``mat_crossc``.
            flux: ``Flux`` provider returning ``(flux_volt, flux_torq)``
                for a given operating point.
            n_theta: Angular resolution for phase-domain sampling. Rounded
                internally to a multiple of ``2 * n_phases`` so that the
                ``[0, 2π]`` electrical period contains an integer number
                of per-phase shift intervals (``2π / n_phases``) at half
                resolution. Required for the future SVPWM zero-sequence
                injection (paper Eq. 9) to compute exact min/max across
                phases, and harmless for the rest of the math.
        """
        self.machine: BaseMachine = machine
        self.flux = flux
        self.n_phases: int = machine.n_phases
        self.dim: int = self.n_phases - 1
        self.omega: float | None = None

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

        self.mat_curr_dq_to_volt_dq_fixed: np.ndarray = self.machine.R_stat.copy()
        self.mat_curr_dq_to_volt_dq_omega: np.ndarray = self.machine.mat_crossc @ self.machine.L_stat

        self._build_phase_mapping()

    @abstractmethod
    def _build_phase_mapping(self) -> None:
        """
        Build the DQ-to-phase mapping. Subclass must set whatever
        attributes ``get_curr_ph`` / ``get_volt_ph`` need (typically
        ``mat_dq_to_ph`` for the healthy case or ``mat_dq_to_ph_all`` for
        the fault cases, plus any null-space row vectors).
        """
        ...

    def _set_omega(self, omega: float) -> None:
        """
        Updates the omega-dependent DQ voltage matrix and its phase-domain
        composition.
        """
        if self.omega is None or omega != self.omega:
            self.omega = omega
            self.mat_curr_dq_to_volt_dq = (
                self.mat_curr_dq_to_volt_dq_fixed + self.omega * self.mat_curr_dq_to_volt_dq_omega
            )
            self._update_phase_voltage_matrix()

    @abstractmethod
    def _update_phase_voltage_matrix(self) -> None:
        """
        Recompose the phase-domain version of the DQ voltage matrix after
        ``mat_curr_dq_to_volt_dq`` has been updated for a new omega.
        """
        ...

    def get_volt_dq(self, omega: float, curr_dq: np.ndarray) -> np.ndarray:
        """
        Calculates the DQ voltage vector ``v_s = U·i_s + u`` (paper Eq. 5
        / 15 / 26 / 37), with ``U = R_s + ω·J·L_s`` and ``u = ω·J·Ψ_PM``.

        ``Ψ_PM`` is a constant machine parameter (current-independent); we
        evaluate the flux provider at ``curr_dq = 0`` to make this
        linearity explicit and to keep the result consistent with
        ``get_volt_map_at_theta``, which also linearizes at zero current.

        Args:
            omega: Electrical speed [rad/s].
            curr_dq: DQ current vector.

        Returns:
            np.ndarray: DQ voltage vector.
        """
        self._set_omega(omega)
        flux_volt, _ = self.flux.get_flux(omega, np.zeros_like(curr_dq))
        vec_volt_bemf_dq = self.omega * self.machine.mat_crossc @ flux_volt
        return self.mat_curr_dq_to_volt_dq @ curr_dq + vec_volt_bemf_dq

    @abstractmethod
    def get_curr_ph(self, omega: float, curr_dq: np.ndarray) -> np.ndarray:
        """
        Transforms DQ currents into phase-domain current time-series.
        Output shape depends on the fault mode.
        """
        ...

    @abstractmethod
    def get_volt_ph(self, omega: float, curr_dq: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns ``(volt_ph, volt_0, volt_raw)`` in the phase domain. Output
        shape depends on the fault mode.
        """
        ...

    def get_extra_constraints(self) -> list[dict[str, Any]]:
        """
        SLSQP-compatible constraint dicts that the optimizer must enforce
        in addition to the current and voltage peak limits.

        Default (healthy, single-fault): empty list.

        Two-fault subclasses override this to return the null-space
        equality ``N @ i_s = 0`` (paper Eq. 23 / 34) that prevents the
        optimizer from inflating components of ``i_s`` that don't
        correspond to physical phase currents.

        This θ-independent form is correct for the **static** single-``i_s``
        scheme (``MotorOptimizer``). For the per-θ scheme use
        ``get_extra_constraints_at_theta``.
        """
        return []

    def get_extra_constraints_at_theta(self, theta_idx: int) -> list[dict[str, Any]]:
        """
        θ-dependent null-space constraint for the **pointwise** scheme.

        Default (healthy, single-fault): empty list. Two-fault subclasses
        override this to return ``N @ R(θ) @ i_s = 0`` (the dynamic form of
        Eq. 23 / 34): ``N`` is orthogonal to ``col(C_red)`` in the αβ frame,
        and the realizable αβ vector at rotor angle θ is ``R(θ) @ i_s``, so
        realizability is ``N @ R(θ) @ i_s = 0``. The static ``N @ i_s = 0``
        only enforces this at θ=0; for a per-θ varying ``i_s`` it leaves the
        realized αβ off ``col(C_red)`` and the dq torque no longer matches the
        realized torque.
        """
        return []

    @abstractmethod
    def get_phase_map_at_theta(self, theta_idx: int) -> np.ndarray:
        """
        Returns the DQ-to-phase mapping matrix at a specific rotor angle index.

        Args:
            theta_idx: Index into ``vec_theta``.

        Returns:
            np.ndarray of shape (n_active_phases, dim): each row maps the DQ
            current vector to the instantaneous current of one active phase.
        """
        ...

    def get_volt_map_at_theta(self, theta_idx: int, omega: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the per-phase voltage mapping at a specific rotor angle.

        The phase voltage is ``v_p = H_volt[p] @ i_s + u_volt[p]``, which
        is exactly ``h_k(θ)·(U·i_s + u)`` from the paper (Eq. 10d / 20d /
        31e / 42e) split into the linear-in-``i_s`` part and the BEMF
        offset. ``u = ω·J·Ψ_PM`` is current-independent, so we evaluate
        the flux provider at ``curr_dq = 0``.

        Args:
            theta_idx: Index into ``vec_theta``.
            omega: Electrical speed [rad/s].

        Returns:
            H_volt: shape (n_active_phases, dim) — linear map from i_s to v_phase.
            u_volt: shape (n_active_phases,) — BEMF offset at this angle.
        """
        self._set_omega(omega)
        flux_volt, _ = self.flux.get_flux(omega, np.zeros(self.dim))
        u_dq = self.omega * self.machine.mat_crossc @ flux_volt
        H_ph = self.get_phase_map_at_theta(theta_idx)      # (n_active_phases, dim)
        H_volt = H_ph @ self.mat_curr_dq_to_volt_dq        # (n_active_phases, dim)
        u_volt = H_ph @ u_dq                               # (n_active_phases,)
        return H_volt, u_volt

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


class Transform(BaseTransform):
    """
    Healthy (fault-free) transform: all phases are operational. The
    phase-domain accessors ``get_curr_ph`` and ``get_volt_ph`` return
    2D arrays of shape ``(n_phases, n_theta + 1)`` covering all 5
    phases (paper Section I uses ``h(θ)·i_s`` per phase; we expose the
    stacked result for every phase so consumers can compute the global
    peak across phases for the constraint ``h(θ)·i_s ≤ I_max ∀θ``).

    Zero-sequence (SVPWM) injection via ``add_volt_0=True`` is defined
    in the paper (Eq. 8–9) but currently disabled in this codebase per
    the user's directive — passing ``add_volt_0=True`` raises
    ``NotImplementedError``. ``add_volt_0=False`` (the default) follows
    paper Eq. 10 with ``v_s0(θ) = 0``.
    """

    def __init__(
        self,
        machine: BaseMachine,
        flux: Flux,
        add_volt_0: bool = False,
        n_theta: int = 700,
    ) -> None:
        self.add_volt_0: bool = add_volt_0
        super().__init__(machine=machine, flux=flux, n_theta=n_theta)

    def _build_phase_mapping(self) -> None:
        """
        Builds the column-stacked sinusoidal basis for harmonics
        1, 3, 5, ..., 2*(dim//2) - 1 (phase-a waveform), plus the
        full all-phase mapping mat_dq_to_ph_all of shape (n_t, n_phases, dim).
        """
        cols = []
        for i in range(self.dim // 2):
            h = 2 * i + 1
            cols.append(np.cos(h * self.vec_theta))
            cols.append(-np.sin(h * self.vec_theta))
        self.mat_dq_to_ph: np.ndarray = np.column_stack(cols)

        n_t = self.vec_theta.size
        phi = 2 * np.pi / self.n_phases
        mat_all = np.zeros((n_t, self.n_phases, self.dim))
        for i in range(self.dim // 2):
            h = 2 * i + 1
            for p in range(self.n_phases):
                # Physical spatial offset for harmonic h at phase p is h*p*phi
                # (the h-th harmonic MMF has h-times the spatial frequency).
                # Note this differs from the standard Concordia matrix's
                # cos(2*angle) third-harmonic row, which aliases 3 <-> -2 at
                # the 5 sample points and yields cos(3*theta - 2*p*phi) on
                # synthesis -- non-physical. The machine parameters are in the
                # physical h*p*phi frame, so we use it directly here.
                theta_arg = h * self.vec_theta - h * p * phi
                mat_all[:, p, 2 * i] = np.cos(theta_arg)
                mat_all[:, p, 2 * i + 1] = -np.sin(theta_arg)
        self.mat_dq_to_ph_all: np.ndarray = mat_all

    def _update_phase_voltage_matrix(self) -> None:
        self.mat_curr_dq_to_volt_ph = np.einsum(
            "tik, kj -> tij", self.mat_dq_to_ph_all, self.mat_curr_dq_to_volt_dq
        )

    def get_phase_map_at_theta(self, theta_idx: int) -> np.ndarray:
        return self.mat_dq_to_ph_all[theta_idx]

    def get_curr_ph(self, omega: float, curr_dq: np.ndarray) -> np.ndarray:
        """
        Returns phase currents at every theta sample for all 5 phases.

        Note: ``omega`` has no effect on the returned current waveform
        (the DQ-to-phase mapping is omega-independent). The argument
        triggers ``_set_omega`` as a side effect, which refreshes the
        omega-dependent voltage matrices that ``get_volt_ph`` later
        reads — callers that interleave current and voltage queries
        rely on this cache being consistent.

        Returns:
            np.ndarray of shape (n_phases, n_theta + 1).
        """
        self._set_omega(omega)
        return np.einsum("tik, k -> it", self.mat_dq_to_ph_all, curr_dq)

    def get_volt_ph(self, omega: float, curr_dq: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns phase voltages at every theta sample for all 5 phases.

        Implements the per-phase voltage ``v_k(θ) = h_k(θ)·v_s`` of the
        paper (Eq. 8 with SVPWM injection ``v_s0`` disabled).  The
        permanent-magnet flux ``Ψ_PM`` is evaluated at ``curr_dq = 0``
        for consistency with ``get_volt_map_at_theta``.

        The 3-tuple return shape ``(volt_ph, volt_0, volt_raw)`` is
        retained for backward compatibility with downstream code that
        anticipated SVPWM injection: with injection disabled,
        ``volt_0`` is zero everywhere and ``volt_ph == volt_raw``.

        Returns:
            Three arrays of shape (n_phases, n_theta + 1).
        """
        self._set_omega(omega)
        flux_volt, _ = self.flux.get_flux(omega, np.zeros_like(curr_dq))
        volt_bemf_dq = self.omega * self.machine.mat_crossc @ flux_volt
        volt_bemf_ph = np.einsum("tik, k -> it", self.mat_dq_to_ph_all, volt_bemf_dq)
        volt_raw = np.einsum("tij, j -> it", self.mat_curr_dq_to_volt_ph, curr_dq) + volt_bemf_ph

        if self.add_volt_0:
            raise NotImplementedError("Zero-sequence (SVPWM) injection not implemented; pass add_volt_0=False.")
        volt_ph = volt_raw
        volt_0 = np.zeros_like(volt_raw)
        return volt_ph, volt_0, volt_raw


def _build_full_clarke_5phase() -> np.ndarray:
    """
    Builds the 5-phase Clarke transform with the (2/5) normalization
    factor pre-applied. Rows are [alpha_1, beta_1, alpha_3, beta_3,
    zero_seq]; columns are phases [a, b, c, d, e] at electrical angles
    0, 2pi/5, 4pi/5, 6pi/5, 8pi/5.

    The third-harmonic rows use ``cos(3*angle)/sin(3*angle)`` rather than
    the textbook Concordia ``cos(2*angle)/sin(2*angle)``. Both have
    identical values at the 5 sample points (3 = -2 mod 5), but only the
    ``cos(3*angle)`` form, composed with the Park ``R(3*theta)`` and the
    pseudoinverse, reconstructs the physical third-harmonic phase current
    ``cos(3*theta - 3*p*phi)``. The ``cos(2*angle)`` form would yield
    ``cos(3*theta - 2*p*phi)``, which is non-physical and inconsistent
    with the healthy ``Transform`` (h*p*phi) and with the machine
    parameters ``L_stat``/``Psi_PM`` (defined in the physical 3*p*phi
    frame).
    """
    C_full = np.zeros((5, 5))
    for p in range(5):
        angle = p * 2 * np.pi / 5
        C_full[0, p] = np.cos(angle)
        C_full[1, p] = np.sin(angle)
        C_full[2, p] = np.cos(3 * angle)
        C_full[3, p] = np.sin(3 * angle)
        C_full[4, p] = 0.5
    C_full *= 2.0 / 5.0
    return C_full


class _BaseTransformFault(BaseTransform):
    """
    Internal scaffolding shared by all fault subclasses. Builds the
    reduced Clarke matrix by removing the zero-sequence row and the
    columns of the open phases, composes per-theta phase mappings via
    einsum, and exposes the resulting 2D phase-domain outputs.

    Subclasses must set ``self._open_phases`` (a sorted tuple of indices
    in [0, 4]) before calling super().__init__.
    """

    _open_phases: tuple[int, ...]

    def _build_phase_mapping(self) -> None:
        if self.n_phases != 5:
            raise NotImplementedError("Fault-tolerant transforms are currently implemented only for 5-phase machines.")

        C_full = _build_full_clarke_5phase()
        kept_phases = [p for p in range(5) if p not in self._open_phases]
        self._kept_phases: tuple[int, ...] = tuple(kept_phases)
        C_red = C_full[:4, :][:, kept_phases]

        if C_red.shape[0] == C_red.shape[1]:
            # Single-fault: C_red is square (4x4) and exactly invertible.
            # No null-space constraint needed (paper Section II).
            self._T_inv: np.ndarray = np.linalg.inv(C_red)
            self._N: np.ndarray | None = None
        else:
            # Two-fault: C_red is rectangular (4x3). Use Moore-Penrose
            # pseudoinverse. The DQ vector is over-parameterized — there is
            # a 1-D left null space of C_red. To pin i_s to the row space
            # (i.e., to physically realizable DQ vectors), the optimizer
            # must enforce N @ i_s = 0 where N spans that left null space
            # (paper Eq. 23 / 34). Without this constraint, the optimizer
            # can inflate null-space components of i_s — producing zero
            # phase currents but non-zero torque per the DQ-frame T(i_s)
            # formula, which is non-physical.
            self._T_inv = np.linalg.pinv(C_red)
            self._N = _compute_null_space_row(C_red)

        n_t = self.vec_theta.size
        R_all = np.zeros((n_t, 4, 4))
        c1, s1 = np.cos(self.vec_theta), np.sin(self.vec_theta)
        c3, s3 = np.cos(3 * self.vec_theta), np.sin(3 * self.vec_theta)
        R_all[:, 0, 0] = c1
        R_all[:, 0, 1] = -s1
        R_all[:, 1, 0] = s1
        R_all[:, 1, 1] = c1
        R_all[:, 2, 2] = c3
        R_all[:, 2, 3] = -s3
        R_all[:, 3, 2] = s3
        R_all[:, 3, 3] = c3

        self.mat_dq_to_ph_all: np.ndarray = np.einsum("ij, tjk -> tik", self._T_inv, R_all)

    def _update_phase_voltage_matrix(self) -> None:
        self.mat_curr_dq_to_volt_ph = np.einsum("tik, kj -> tij", self.mat_dq_to_ph_all, self.mat_curr_dq_to_volt_dq)

    def get_phase_map_at_theta(self, theta_idx: int) -> np.ndarray:
        return self.mat_dq_to_ph_all[theta_idx]

    def get_extra_constraints(self) -> list[dict[str, Any]]:
        """
        For two-fault modes, returns the null-space equality constraint
        ``N @ i_s = 0`` (paper Eq. 23 / 34). For single-fault mode, returns
        an empty list (no over-parameterization, no constraint needed).
        """
        if self._N is None:
            return []
        N = self._N
        return [{"type": "eq", "fun": lambda i_s, N=N: float(N @ i_s)}]

    def get_extra_constraints_at_theta(self, theta_idx: int) -> list[dict[str, Any]]:
        """
        θ-dependent null-space equality ``N @ R(θ) @ i_s = 0`` for the
        per-θ (pointwise) scheme. Pins the αβ vector ``R(θ) @ i_s`` to
        ``col(C_red)`` so the realized phase currents match ``i_s`` and the
        dq torque equals the realized torque. Single-fault returns an empty
        list (square ``C_red``, no null space).
        """
        if self._N is None:
            return []
        theta = float(self.vec_theta[theta_idx])
        c1, s1 = np.cos(theta), np.sin(theta)
        c3, s3 = np.cos(3 * theta), np.sin(3 * theta)
        R = np.array(
            [
                [c1, -s1, 0, 0],
                [s1, c1, 0, 0],
                [0, 0, c3, -s3],
                [0, 0, s3, c3],
            ]
        )
        NR = self._N @ R
        return [{"type": "eq", "fun": lambda i_s, NR=NR: float(NR @ i_s)}]

    def get_curr_ph(self, omega: float, curr_dq: np.ndarray) -> np.ndarray:
        """
        Returns phase currents at every theta sample.

        Note: ``omega`` has no effect on the returned current waveform
        (phase currents derive only from the DQ-to-phase mapping, which is
        omega-independent). The argument is retained so that ``_set_omega``
        is invoked as a side effect, refreshing the omega-dependent
        voltage matrix cache that ``get_volt_ph`` relies on.

        Returns:
            np.ndarray of shape (n_healthy_phases, n_theta + 1).
        """
        self._set_omega(omega)
        return np.einsum("tik, k -> it", self.mat_dq_to_ph_all, curr_dq)

    def get_volt_ph(self, omega: float, curr_dq: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns phase voltages at every theta sample for the active
        (healthy) phases. Implements ``v_k(θ) = h_k(θ)·v_s`` from the
        paper (Eq. 18 / 29 / 40), with zero-sequence injection disabled
        per the fault-mode model (Eq. 19 / 30 / 41).

        The 3-tuple return shape is kept symmetric with the healthy
        ``Transform.get_volt_ph``; ``volt_0`` is always zero and
        ``volt_ph == volt_raw``.

        Returns:
            Three arrays of shape (n_healthy_phases, n_theta + 1).
        """
        self._set_omega(omega)
        flux_volt, _ = self.flux.get_flux(omega, np.zeros_like(curr_dq))
        volt_bemf_dq = self.omega * self.machine.mat_crossc @ flux_volt

        volt_bemf_ph = np.einsum("tik, k -> it", self.mat_dq_to_ph_all, volt_bemf_dq)
        volt_raw = np.einsum("tij, j -> it", self.mat_curr_dq_to_volt_ph, curr_dq) + volt_bemf_ph

        volt_ph = volt_raw
        volt_0 = np.zeros_like(volt_raw)
        return volt_ph, volt_0, volt_raw


class TransformFault1(_BaseTransformFault):
    """
    Single open-phase fault (paper Section II). The healthy phases form
    a 4-phase system whose reduced Clarke matrix is square (4x4) and
    directly invertible; the 4-D DQ vector maps bijectively to physical
    phase currents, so no null-space constraint is needed and
    ``get_extra_constraints`` returns an empty list.

    Phase indexing (5-phase machine):
        0 -> 'a' at angle 0
        1 -> 'b' at angle 2*pi/5
        2 -> 'c' at angle 4*pi/5
        3 -> 'd' at angle 6*pi/5
        4 -> 'e' at angle 8*pi/5
    """

    def __init__(
        self,
        machine: BaseMachine,
        flux: Flux,
        open_phase: int,
        n_theta: int = 700,
    ) -> None:
        """
        Args:
            machine: 5-phase machine.
            flux: Flux provider.
            open_phase: Index of the open phase in [0, 4].
            n_theta: Angular resolution.
        """
        if not isinstance(open_phase, (int, np.integer)) or not (0 <= open_phase <= 4):
            raise ValueError(f"open_phase must be an int in [0, 4]; got {open_phase!r}.")
        self._open_phases = (int(open_phase),)
        super().__init__(machine=machine, flux=flux, n_theta=n_theta)


class TransformFault2Adjacent(_BaseTransformFault):
    """
    Two adjacent open-phase fault (paper Section III). The reduced Clarke
    matrix is rectangular (4 rows, 3 columns) so the inverse is replaced
    by the Moore-Penrose pseudoinverse. The 4-D DQ vector is therefore
    over-parameterized with respect to the 3-D space of realizable phase
    currents, and the optimizer enforces the null-space equality
    ``N @ i_s = 0`` (paper Eq. 23) to pin ``i_s`` to the row space of the
    reduced Clarke. This is delivered automatically via
    ``get_extra_constraints``.

    Two phases are considered adjacent if their indices differ by 1 on
    the modular ring of size 5 (so {0,1}, {1,2}, {2,3}, {3,4}, and
    {4,0} are all adjacent; phase 'e' at index 4 is adjacent to phase
    'a' at index 0).
    """

    def __init__(
        self,
        machine: BaseMachine,
        flux: Flux,
        open_phases: tuple[int, int],
        n_theta: int = 700,
    ) -> None:
        """
        Args:
            open_phases: Two distinct indices in [0, 4] that are
                adjacent on the 5-phase ring.
        """
        _validate_two_phase_indices(open_phases, required_kind="adjacent")
        self._open_phases = tuple(sorted(int(p) for p in open_phases))
        super().__init__(machine=machine, flux=flux, n_theta=n_theta)


class TransformFault2NonAdjacent(_BaseTransformFault):
    """
    Two non-adjacent open-phase fault (paper Section IV). Structurally
    identical to ``TransformFault2Adjacent``: rectangular reduced Clarke
    matrix, pseudoinverse, and the null-space equality
    ``N @ i_s = 0`` (paper Eq. 34) enforced via ``get_extra_constraints``.
    Only the validation rule on ``open_phases`` differs.
    """

    def __init__(
        self,
        machine: BaseMachine,
        flux: Flux,
        open_phases: tuple[int, int],
        n_theta: int = 700,
    ) -> None:
        """
        Args:
            open_phases: Two distinct indices in [0, 4] that are
                non-adjacent on the 5-phase ring (ring distance 2 or 3).
        """
        _validate_two_phase_indices(open_phases, required_kind="non_adjacent")
        self._open_phases = tuple(sorted(int(p) for p in open_phases))
        super().__init__(machine=machine, flux=flux, n_theta=n_theta)


def _validate_two_phase_indices(open_phases: tuple[int, int], required_kind: str) -> None:
    """
    Validates that ``open_phases`` is a 2-tuple of distinct indices in
    [0, 4] and that their ring distance matches the requested kind.
    """
    if len(open_phases) != 2:
        raise ValueError(f"open_phases must contain exactly two indices; got {open_phases!r}.")
    a, b = int(open_phases[0]), int(open_phases[1])
    if not (0 <= a <= 4 and 0 <= b <= 4):
        raise ValueError(f"open_phases indices must lie in [0, 4]; got {open_phases!r}.")
    if a == b:
        raise ValueError("open_phases must be two distinct indices.")
    ring_dist = min((a - b) % 5, (b - a) % 5)
    if required_kind == "adjacent" and ring_dist != 1:
        raise ValueError(
            f"open_phases {open_phases!r} are not adjacent on the 5-phase ring "
            f"(ring distance is {ring_dist}, expected 1). Use "
            f"TransformFault2NonAdjacent instead."
        )
    if required_kind == "non_adjacent" and ring_dist not in (2,):
        raise ValueError(
            f"open_phases {open_phases!r} are not non-adjacent on the 5-phase "
            f"ring (ring distance is {ring_dist}, expected 2). Use "
            f"TransformFault2Adjacent instead."
        )


def _compute_null_space_row(C_red: np.ndarray) -> np.ndarray:
    """
    Returns the (single) row vector spanning the left null space of
    ``C_red`` (shape (4, 3) for the two-fault case). The vector is
    normalized to unit length so the null-space constraint
    ``N @ curr_dq == 0`` has a numerically uniform scale across modes.
    """
    _, s, vh = np.linalg.svd(C_red.T)
    null_row = vh[-1, :]
    null_row = null_row / np.linalg.norm(null_row)
    return null_row.reshape(1, -1)[0]
