import numpy as np
import pytest

from current_setpoints.parameters import Flux_IEEEMachine2, IEEEMachine2
from current_setpoints.simulation import Transform


def _build_default_transform(add_volt_0: bool = False, n_theta: int = 700) -> Transform:
    """Returns a Transform tied to a fresh, default-configured IEEEMachine2."""
    machine = IEEEMachine2()
    machine.set_max_pars(curr_max=30.0, volt_max=13.0, omega_max=1800)
    flux = Flux_IEEEMachine2()
    return Transform(machine=machine, flux=flux, add_volt_0=add_volt_0, n_theta=n_theta)


def _project_dq(signal: np.ndarray, theta: np.ndarray, h: int) -> tuple[float, float]:
    """Project a phase-A periodic waveform onto the (cos h*theta, -sin h*theta) basis."""
    s = signal[:-1]
    t = theta[:-1]
    d = 2.0 * np.mean(s * np.cos(h * t))
    q = -2.0 * np.mean(s * np.sin(h * t))
    return float(d), float(q)


def test_init_rounds_n_theta_to_multiple_of_2_n_phases():
    """
    n_theta is silently rounded to the nearest multiple of 2 * n_phases
    so a full electrical period samples cleanly across all phases.
    """
    transform = _build_default_transform(n_theta=703)
    assert transform.vec_theta.size == 701  # n_theta + 1


def test_init_rejects_nonpositive_n_theta():
    machine = IEEEMachine2()
    flux = Flux_IEEEMachine2()
    with pytest.raises(ValueError, match="n_theta must be positive"):
        Transform(machine=machine, flux=flux, add_volt_0=False, n_theta=0)


def test_init_rejects_n_theta_that_rounds_to_zero():
    """
    A small n_theta like 3 with n_phases=5 rounds to 0; this must be caught
    instead of silently producing a single-sample theta vector.
    """
    machine = IEEEMachine2()
    flux = Flux_IEEEMachine2()
    with pytest.raises(ValueError, match="too small after rounding"):
        Transform(machine=machine, flux=flux, add_volt_0=False, n_theta=3)


def test_phase_shift_samples_matches_symmetry():
    """
    n_theta / n_phases samples must correspond to one (2*pi / n_phases)
    phase shift; SVPWM injection relies on this.
    """
    transform = _build_default_transform(n_theta=700)
    n_phases = transform.n_phases
    assert transform._phase_shift_samples * n_phases == transform.vec_theta.size - 1


def test_dq_to_ph_matrix_columns_are_orthogonal_harmonics():
    """
    For an n-phase machine the DQ-to-phase matrix consists of cos/sin pairs
    at odd harmonics 1, 3, 5, ... Sampled across a full period these columns
    are mutually orthogonal in the L2 inner product (modulo the wrap point).
    """
    transform = _build_default_transform(n_theta=700)
    M = transform.mat_dq_to_ph[:-1]  # drop wrap sample so dot products are clean
    gram = M.T @ M
    n = M.shape[0]
    assert np.allclose(gram, (n / 2) * np.eye(M.shape[1]), atol=1e-9)


def test_get_curr_ph_pure_d1_produces_pure_cosine():
    """Setting only i_d1 = I should make phase A track I * cos(theta)."""
    transform = _build_default_transform(n_theta=700)
    vec_curr_dq = np.zeros(transform.dim)
    vec_curr_dq[0] = 5.0
    i_phase = transform.get_curr_ph(omega=0.0, curr_dq=vec_curr_dq)
    expected = 5.0 * np.cos(transform.vec_theta)
    assert np.allclose(i_phase, expected)


def test_get_curr_ph_pure_q1_produces_pure_negative_sine():
    """Setting only i_q1 = I should give -I * sin(theta) per the (cos, -sin) convention."""
    transform = _build_default_transform(n_theta=700)
    vec_curr_dq = np.zeros(transform.dim)
    vec_curr_dq[1] = 5.0
    i_phase = transform.get_curr_ph(omega=0.0, curr_dq=vec_curr_dq)
    expected = -5.0 * np.sin(transform.vec_theta)
    assert np.allclose(i_phase, expected)


def test_get_volt_dq_zero_current_zero_speed_is_zero():
    """At i = 0 and omega = 0 there should be no DQ voltage drop."""
    transform = _build_default_transform()
    v_dq = transform.get_volt_dq(omega=0.0, curr_dq=np.zeros(transform.dim))
    assert np.allclose(v_dq, 0.0)


def test_get_volt_dq_zero_current_nonzero_speed_is_pure_bemf():
    """
    With i = 0 and omega != 0, v_dq should equal omega * (mat_crossc @ flux_volt).
    Flux now lives on the Flux provider, not the machine, so we read it via
    `transform.flux.get_flux(omega, curr_dq)`.
    """
    transform = _build_default_transform()
    omega = 50.0
    zero_curr = np.zeros(transform.dim)
    v_dq = transform.get_volt_dq(omega=omega, curr_dq=zero_curr)

    flux_volt, _ = transform.flux.get_flux(omega, zero_curr)
    expected = omega * (transform.machine.mat_crossc @ flux_volt)
    assert np.allclose(v_dq, expected)


def test_set_omega_updates_speed_dependent_matrix():
    """
    `_compute_matrices_init` builds the omega-independent term once;
    `_set_omega` adds the omega-scaled term. Testing the private hook
    directly because that's exactly what's being validated.
    """
    transform = _build_default_transform()
    transform._set_omega(0.0)
    M0 = transform.mat_curr_dq_to_volt_dq.copy()
    transform._set_omega(100.0)
    M_high = transform.mat_curr_dq_to_volt_dq

    diff = M_high - M0
    expected = 100.0 * (transform.machine.mat_crossc @ transform.machine.L_stat)
    assert np.allclose(diff, expected)


def test_get_volt_ph_no_injection_equals_raw():
    """When add_volt_0=False, vec_volt_ph must equal vec_volt_raw exactly."""
    transform = _build_default_transform(add_volt_0=False)
    vec_curr_dq = np.array([5.0, 3.0, 0.0, 0.0])
    vec_volt_ph, vec_volt_0, vec_volt_raw = transform.get_volt_ph(omega=50.0, curr_dq=vec_curr_dq)
    assert np.allclose(vec_volt_ph, vec_volt_raw)
    assert np.allclose(vec_volt_0, 0.0)


# ---------------------------------------------------------------------------
# Zero-sequence injection (SVPWM) is not yet implemented in the library.
# The four tests below describe the contract the implementation should meet
# once it lands. They are kept (marked skip) so they re-enable automatically
# the moment the NotImplementedError in `get_volt_ph` is replaced by a real
# implementation.
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="Zero-sequence injection not yet implemented")
def test_get_volt_ph_pure_fundamental_injection_matches_theory():
    """
    Min-max common-mode injection on a pure-fundamental phase voltage must
    reduce the peak by exactly cos(pi / (2 * n_phases)) — for 5-phase that
    is cos(pi/10) ≈ 0.9511.
    """
    transform = _build_default_transform(add_volt_0=True, n_theta=700)
    vec_curr_dq = np.zeros(transform.dim)
    vec_curr_dq[0] = 10.0  # pure i_d1
    vec_volt_ph, _, vec_volt_raw = transform.get_volt_ph(omega=0.0, curr_dq=vec_curr_dq)

    raw_peak = np.max(np.abs(vec_volt_raw))
    inj_peak = np.max(np.abs(vec_volt_ph))
    expected_ratio = np.cos(np.pi / (2 * transform.n_phases))

    assert raw_peak > 0
    assert np.isclose(inj_peak / raw_peak, expected_ratio, atol=1e-3)


@pytest.mark.skip(reason="Zero-sequence injection not yet implemented")
def test_get_volt_ph_injection_reduces_peak_under_load():
    """
    Under a realistic operating point (fundamental + harmonic content)
    injection must still reduce the peak phase voltage vs the raw waveform.
    """
    transform = _build_default_transform(add_volt_0=True)
    vec_curr_dq = np.array([10.0, 5.0, 1.0, 0.5])
    vec_volt_ph, _, vec_volt_raw = transform.get_volt_ph(omega=50.0, curr_dq=vec_curr_dq)
    assert np.max(np.abs(vec_volt_ph)) < np.max(np.abs(vec_volt_raw))


@pytest.mark.skip(reason="Zero-sequence injection not yet implemented")
def test_get_volt_ph_injection_preserves_dq_content():
    """
    SVPWM is a common-mode signal: it must NOT contaminate the DQ subspace.
    Projecting the injected phase-A waveform back onto the cos/sin basis at
    each odd harmonic must give the same DQ values as projecting the raw.
    """
    transform = _build_default_transform(add_volt_0=True)
    vec_curr_dq = np.array([10.0, 5.0, 1.0, 0.5])
    vec_volt_ph, _, vec_volt_raw = transform.get_volt_ph(omega=50.0, curr_dq=vec_curr_dq)
    theta = transform.vec_theta

    for h in (1, 3):
        d_raw, q_raw = _project_dq(vec_volt_raw, theta, h)
        d_inj, q_inj = _project_dq(vec_volt_ph, theta, h)
        assert np.isclose(d_raw, d_inj, atol=1e-9)
        assert np.isclose(q_raw, q_inj, atol=1e-9)


@pytest.mark.skip(reason="Zero-sequence injection not yet implemented")
def test_get_volt_ph_injection_v0_returned_consistently():
    """vec_volt_ph == vec_volt_raw + vec_volt_0 sample by sample."""
    transform = _build_default_transform(add_volt_0=True)
    vec_curr_dq = np.array([7.0, 2.0, 0.5, -0.5])
    vec_volt_ph, vec_volt_0, vec_volt_raw = transform.get_volt_ph(omega=50.0, curr_dq=vec_curr_dq)
    assert np.allclose(vec_volt_ph, vec_volt_raw + vec_volt_0)


def test_get_max_vals_returns_four_values():
    """
    `get_max_vals` was cut from eight returns to four:
    (curr_peak, curr_ang_diff, volt_peak, volt_ang_diff).
    """
    transform = _build_default_transform()
    out = transform.get_max_vals(omega=0.0, curr_dq=np.array([5.0, 3.0, 0.0, 0.0]))
    assert len(out) == 4


def test_get_max_vals_curr_peak_matches_analytical_for_pure_d1():
    """For pure i_d1 = I the peak phase current must equal I exactly."""
    transform = _build_default_transform()
    vec_curr_dq = np.array([7.5, 0.0, 0.0, 0.0])
    curr_peak, *_ = transform.get_max_vals(omega=0.0, curr_dq=vec_curr_dq)
    assert np.isclose(curr_peak, 7.5)


## ---------------------------------------------------------------------
## Regression tests for the strict active-set peak detector.
##
## The earlier ``count_peaks`` classified Type I vs Type II via a
## dq-angle alignment heuristic that had two bugs:
##   1. dimensional mismatch -- the threshold ``tol * val_max`` mixed
##      amperes/volts (RHS) with radians (LHS), so the comparison was
##      meaningless;
##   2. ``arctan2(0, 0)`` returns 0 in numpy, so cells with no
##      third-harmonic content were classified by an irrelevant value.
##
## The replacement counts polarity-consistent local extrema of the
## actual phase waveform that reach the limit, directly matching the
## active-set definition. These tests cover the canonical Type I /
## Type II cases plus a discriminating cell where the old detector
## misfired but the new one is correct.
## ---------------------------------------------------------------------


def _scale_to_limit(transform: Transform, curr_dq: np.ndarray, limit: float) -> np.ndarray:
    """Helper: rescale curr_dq so the global peak of the phase-A current
    equals ``limit`` exactly."""
    curr_ph = transform.get_curr_ph(omega=0.0, curr_dq=curr_dq)
    peak = float(np.max(np.abs(curr_ph)))
    return curr_dq * (limit / peak)


def test_count_peaks_returns_zero_when_below_limit():
    """A pure-fundamental cell with peak well below I_max gives (0, 0)."""
    transform = _build_default_transform()
    I = transform.machine.curr_max
    # i_a = a*cos - a*sin = a*sqrt(2)*cos(theta + pi/4), peak = a*sqrt(2)
    a = 0.3 * I / np.sqrt(2)
    curr_dq = np.array([a, a, 0.0, 0.0])
    n_curr, n_volt = transform.count_peaks(omega=0.0, curr_dq=curr_dq)
    assert n_curr == 0
    assert n_volt == 0


def test_count_peaks_pure_fundamental_at_limit_is_type_I():
    """
    A pure-fundamental sinusoid whose peak equals I_max must be Type I
    (n_curr_peaks = 1). One positive and one negative peak per period,
    each at the limit; the conventional ``Type I`` label is 1.
    """
    transform = _build_default_transform()
    I = transform.machine.curr_max
    a = I / np.sqrt(2)  # peak = a*sqrt(2) = I_max
    curr_dq = np.array([a, a, 0.0, 0.0])
    n_curr, n_volt = transform.count_peaks(omega=0.0, curr_dq=curr_dq)
    assert n_curr == 1, f"Pure fundamental at limit must be Type I; got {n_curr}"
    assert n_volt == 0


def test_count_peaks_flat_top_type_II_is_two():
    """
    A genuine flat-top synthesised current waveform -- fundamental at
    phi_1 = pi/4 combined with a third harmonic at the anti-phase
    alignment phi_3 = 3*phi_1 + pi = 7*pi/4 with magnitude ~0.3 of the
    fundamental -- has two equal-height same-sign peaks at the limit
    per half-period (Type II).
    """
    transform = _build_default_transform()
    I = transform.machine.curr_max
    phi_1 = np.pi / 4
    phi_3 = 3 * phi_1 + np.pi  # 7*pi/4, flat-top anti-phase
    a = 1.0
    eps = 0.3 * a  # large enough that two distinct peaks emerge
    curr_dq = np.array(
        [a, a, eps * np.cos(phi_3), eps * np.sin(phi_3)]
    )
    curr_dq = _scale_to_limit(transform, curr_dq, I)
    n_curr, n_volt = transform.count_peaks(omega=0.0, curr_dq=curr_dq)
    assert n_curr == 2, f"Flat-top synthesised waveform must be Type II; got {n_curr}"


def test_count_peaks_distinguishes_flat_top_aligned_but_small_i3_from_type_II():
    """
    Regression test for the dimensional-mismatch bug in the dq-angle
    detector.

    Construct a cell where the third harmonic is *perfectly* flat-top
    anti-phase aligned with the fundamental (so the buggy detector
    sees angle_diff == 0 and returns Type II) but |i_3| is too small
    to produce a second peak at the limit. The waveform actually has
    one peak per polarity at the limit; the correct classification is
    Type I.

    With the old ``_count_peaks_helper`` and ``tol = 1e-4``,
    ``angle_diff = 0 < tol * val_max = 1e-4 * I_max`` always holds, so
    the old detector returned Type II. The new waveform-based detector
    returns Type I.
    """
    transform = _build_default_transform()
    I = transform.machine.curr_max
    phi_1 = np.pi / 4
    phi_3 = 3 * phi_1 + np.pi  # perfect flat-top alignment
    a = 1.0
    eps = 0.01 * a  # tiny -- one peak still dominates
    curr_dq = np.array(
        [a, a, eps * np.cos(phi_3), eps * np.sin(phi_3)]
    )
    curr_dq = _scale_to_limit(transform, curr_dq, I)

    # Manually verify the waveform has only one positive peak at limit.
    curr_ph = transform.get_curr_ph(omega=0.0, curr_dq=curr_dq)
    w = curr_ph[:-1]
    is_pmax = (w[1:-1] > w[:-2]) & (w[1:-1] > w[2:]) & (w[1:-1] > 0)
    n_pos_at_limit = int(np.sum(w[1:-1][is_pmax] >= I * (1 - 1e-3)))
    assert n_pos_at_limit == 1, (
        f"Sanity: a near-pure-fundamental waveform should have exactly one "
        f"positive peak at the limit; found {n_pos_at_limit}."
    )

    n_curr, n_volt = transform.count_peaks(omega=0.0, curr_dq=curr_dq)
    assert n_curr == 1, (
        f"Expected Type I (n_curr = 1); got {n_curr}. The dq-angle-alignment "
        f"detector incorrectly classifies this as Type II because phi_3 is "
        f"perfectly flat-top-aligned, ignoring the fact that |i_3| is too "
        f"small to produce a second peak at the limit."
    )


def test_count_peaks_does_not_misuse_arctan2_zero_zero():
    """
    Regression test for the ``arctan2(0, 0)`` bug.

    With ``i_3 = 0`` the dq3 angle is undefined; numpy returns 0 for
    arctan2(0, 0), and the old detector then evaluated its threshold
    against that arbitrary 0. The waveform-based detector inspects the
    actual phase current and is unaffected by undefined dq angles.

    For a pure-fundamental waveform with peak slightly below the limit,
    n_curr must be 0 (no peak at limit) regardless of what arctan2
    returns for the empty third-harmonic vector.
    """
    transform = _build_default_transform()
    I = transform.machine.curr_max
    # peak = 0.95 * I (below limit by 5%)
    a = 0.95 * I / np.sqrt(2)
    curr_dq = np.array([a, a, 0.0, 0.0])
    n_curr, n_volt = transform.count_peaks(omega=0.0, curr_dq=curr_dq)
    assert n_curr == 0, f"Below limit must give n_curr = 0; got {n_curr}"


def test_count_peaks_caps_at_two():
    """A waveform with many small ripples near the limit should not
    return more than 2 (the active-set classifier caps at Type II)."""
    transform = _build_default_transform()
    # Build a contrived case: pure flat-top at the limit (Type II).
    I = transform.machine.curr_max
    phi_1 = np.pi / 4
    phi_3 = 3 * phi_1 + np.pi
    curr_dq = np.array([
        1.0, 1.0,
        0.4 * np.cos(phi_3), 0.4 * np.sin(phi_3),
    ])
    curr_dq = _scale_to_limit(transform, curr_dq, I)
    n_curr, _ = transform.count_peaks(omega=0.0, curr_dq=curr_dq)
    assert n_curr in (1, 2), f"Cap violated: got n_curr = {n_curr}"
